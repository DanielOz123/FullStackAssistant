import boto3
import json
import math
from datetime import datetime
import os
import time
from collections import defaultdict

# Initialize clients
session = boto3.Session()
dynamodb = session.resource('dynamodb')
bedrock = session.client('bedrock-runtime')

def get_titan_embeddings(text):
    """Generate embeddings using Amazon Titan"""
    try:
        if len(text) > 10000:
            text = text[:10000]
            
        response = bedrock.invoke_model(
            modelId='amazon.titan-embed-text-v2:0',
            body=json.dumps({'inputText': text})
        )
        response_body = json.loads(response['body'].read())
        return response_body['embedding']
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        raise

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity without numpy"""
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(x * x for x in vec1))
    magnitude2 = math.sqrt(sum(x * x for x in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def search_similar_chunks_balanced(query_embedding, limit=12):
    """Search for similar chunks with balanced representation from different documents"""
    table = dynamodb.Table(os.environ['EMBEDDINGS_TABLE'])
    
    # Scan all items with pagination
    items = []
    last_evaluated_key = None
    
    while True:
        if last_evaluated_key:
            response = table.scan(ExclusiveStartKey=last_evaluated_key)
        else:
            response = table.scan()
        
        items.extend(response.get('Items', []))
        last_evaluated_key = response.get('LastEvaluatedKey')
        if not last_evaluated_key:
            break
    
    print(f"Total chunks in database: {len(items)}")
    
    # Group items by source file
    items_by_source = defaultdict(list)
    for item in items:
        source_file = item.get('source_file', 'unknown')
        items_by_source[source_file].append(item)
    
    print(f"Documents found: {list(items_by_source.keys())}")
    
    # Calculate similarities for all chunks
    all_similarities = []
    for item in items:
        try:
            chunk_embedding = json.loads(item['embedding'])
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            
            # Bonus for longer chunks (more context)
            content_length = len(item.get('content', ''))
            length_bonus = min(content_length / 800, 0.15)  # Max 15% bonus
            
            final_score = similarity + length_bonus
            all_similarities.append((final_score, item))
            
        except Exception as e:
            print(f"Error processing item: {str(e)}")
            continue
    
    # Sort all chunks by score
    all_similarities.sort(key=lambda x: x[0], reverse=True)
    
    # Strategy 1: Take top chunks from each document
    selected_chunks = []
    chunks_per_document = max(2, limit // len(items_by_source)) if items_by_source else limit
    
    for source_file, source_items in items_by_source.items():
        # Get top chunks for this document
        doc_similarities = [(score, item) for score, item in all_similarities if item.get('source_file') == source_file]
        doc_similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Take top chunks from this document
        top_doc_chunks = doc_similarities[:chunks_per_document]
        selected_chunks.extend(top_doc_chunks)
        print(f"Document {source_file}: selected {len(top_doc_chunks)} chunks")
    
    # Strategy 2: Add overall top chunks to ensure quality
    overall_top_chunks = all_similarities[:min(6, limit)]
    selected_chunks.extend(overall_top_chunks)
    
    # Remove duplicates and get unique chunks
    unique_chunks = {}
    for score, item in selected_chunks:
        chunk_id = f"{item.get('document_id')}_{item.get('chunk_id')}"
        if chunk_id not in unique_chunks or score > unique_chunks[chunk_id][0]:
            unique_chunks[chunk_id] = (score, item)
    
    # Convert back to list and sort
    final_chunks = sorted(unique_chunks.values(), key=lambda x: x[0], reverse=True)
    
    # Take top limit chunks
    result_chunks = [item for score, item in final_chunks[:limit]]
    
    print(f"Selected {len(result_chunks)} chunks from {len(unique_chunks)} unique chunks")
    
    # Log selection details
    source_count = defaultdict(int)
    for chunk in result_chunks:
        source_file = chunk.get('source_file', 'unknown')
        source_count[source_file] += 1
    
    print("Final chunk distribution:")
    for source, count in source_count.items():
        print(f"  {source}: {count} chunks")
    
    return result_chunks

def invoke_claude_3(prompt):
    """Invoke Claude 3 model"""
    CLAUDE_MODEL = 'anthropic.claude-3-sonnet-20240229-v1:0'
    # CLAUDE_MODEL = 'anthropic.claude-3-5-haiku-v1:0'
    try:
        response = bedrock.invoke_model(
            modelId=CLAUDE_MODEL,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1500,  # Aumentar tokens para respuestas más completas
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    except Exception as e:
        print(f"Error invoking Claude 3: {str(e)}")
        return f"Error: {str(e)}"

def lambda_handler(event, context):
    try:
        print(f"Event received: {json.dumps(event)}")
        
        # Handle API Gateway event
        if 'queryStringParameters' in event and event['queryStringParameters']:
            query = event['queryStringParameters'].get('question', '')
        elif 'body' in event and event['body']:
            body = json.loads(event['body'])
            query = body.get('question', '')
        else:
            query = 'Hello'
        
        if not query:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'No question provided'})
            }
        
        print(f"Processing query: {query}")
        
        # Generate embedding for the query
        query_embedding = get_titan_embeddings(query)
        print(f"Query embedding generated: {len(query_embedding)} dimensions")
        
        # Search for similar chunks with balanced approach
        similar_chunks = search_similar_chunks_balanced(query_embedding, limit=10)
        
        if not similar_chunks:
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'answer': 'No relevant information found in the documents.',
                    'sources': []
                })
            }
        
        # Log found chunks
        print(f"Found {len(similar_chunks)} relevant chunks from {len(set(chunk['source_file'] for chunk in similar_chunks))} documents")
        
        # Build context from similar chunks, grouped by document
        context_parts = []
        current_document = None
        
        for chunk in similar_chunks:
            source_file = chunk['source_file']
            if source_file != current_document:
                context_parts.append(f"\n--- Document: {source_file} ---")
                current_document = source_file
            context_parts.append(chunk['content'])
        
        context = "\n".join(context_parts)
        
        # Create prompt that encourages using multiple sources
        prompt_template = """Eres un asistente experto que analiza información de múltiples documentos. 

INSTRUCCIONES IMPORTANTES:
1. Analiza TODOS los documentos proporcionados en el contexto
2. Combina la información relevante de TODAS las fuentes disponibles
3. Si diferentes documentos tienen información complementaria, intégrala
4. Si hay información contradictoria entre documentos, menciónalo
5. Responde en el mismo idioma de la pregunta

CONTEXTO DE MÚLTIPLES DOCUMENTOS:
{context}

PREGUNTA: {question}

Responde de manera completa integrando la información de TODOS los documentos relevantes. Si usas información de un documento específico, menciónalo brevemente."""
        
        prompt = prompt_template.format(context=context, question=query)
        
        # Get response from Claude 3
        response = invoke_claude_3(prompt)
        
        # Get unique sources
        sources = list(set([chunk['source_file'] for chunk in similar_chunks]))
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'answer': response,
                'sources': sources,
                'context_chunks': len(similar_chunks),
                'documents_used': len(sources)
            })
        }
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': f'Error processing query: {str(e)}'})
        }