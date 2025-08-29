import boto3
import json
import math
from datetime import datetime
import os
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

# Initialize AWS clients with session for better resource management and connection pooling
session = boto3.Session()
dynamodb = session.resource('dynamodb')  # For DynamoDB operations (table scans, queries)
bedrock = session.client('bedrock-runtime')  # For Amazon Bedrock model invocations

# Constants for model configuration and performance optimization
TITAN_EMBED_MODEL = 'amazon.titan-embed-text-v2:0'
DEFAULT_CHUNK_LIMIT = 15
MAX_QUERY_LENGTH = 10000  # Characters (safety limit for embedding generation)

def get_titan_embeddings(text: str) -> List[float]:
    """
    Generate text embeddings using Amazon Titan Embeddings model.
    
    Converts input text into a high-dimensional vector representation (1536 dimensions)
    that captures semantic meaning for similarity search and retrieval.
    
    Args:
        text (str): Input text to generate embeddings for. Automatically truncated 
                   to 10,000 characters to avoid token limits and manage costs.
        
    Returns:
        List[float]: 1536-dimensional embedding vector representing semantic meaning
        
    Raises:
        Exception: If embedding generation fails due to model errors, API limits,
                  or network issues. Exception is re-raised for upstream handling.
        
    Example:
        embedding = get_titan_embeddings("What are the company policies?")
        len(embedding)  # Returns 1536
        
    """
    try:
        # Truncate very long texts to avoid token limits and manage API costs
        # Titan Embed v2 supports ~8,000 tokens, so we conservatively limit to 10,000 chars
        if len(text) > MAX_QUERY_LENGTH:
            text = text[:MAX_QUERY_LENGTH]
            print(f"Text truncated to {MAX_QUERY_LENGTH} characters for embeddings")
            
        # Invoke Amazon Titan Embeddings model v2
        response = bedrock.invoke_model(
            modelId=TITAN_EMBED_MODEL,
            body=json.dumps({'inputText': text})
        )
        
        # Parse and return the embedding vector from response
        response_body = json.loads(response['body'].read())
        return response_body['embedding']
        
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        # Re-raise to allow upstream error handling and proper logging
        raise

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Cosine similarity measures the cosine of the angle between two vectors in
    multidimensional space, providing a value between -1 and 1 where:
    - 1: Vectors are identical in direction
    - 0: Vectors are orthogonal (no similarity)
    - -1: Vectors are opposite in direction
    
    This is the preferred metric for semantic similarity search in vector space
    because it's magnitude-invariant and focuses on directional similarity.
    
    Args:
        vec1 (List[float]): First vector (typically query embedding)
        vec2 (List[float]): Second vector (typically chunk embedding)
        
    Returns:
        float: Similarity score between 0 (no similarity) and 1 (identical direction)
        
        
    Mathematical Formula:
        similarity = (A · B) / (||A|| * ||B||)
        where A · B is dot product, ||A|| is magnitude of A
    """
    # Calculate dot product (sum of element-wise multiplication)
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    
    # Calculate magnitudes (Euclidean norms) - L2 norm
    magnitude1 = math.sqrt(sum(x * x for x in vec1))
    magnitude2 = math.sqrt(sum(x * x for x in vec2))
    
    # Handle zero vectors to avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
        
    # Return cosine similarity (dot product normalized by magnitudes)
    return dot_product / (magnitude1 * magnitude2)

def search_similar_chunks_balanced(query_embedding: List[float], limit: int = DEFAULT_CHUNK_LIMIT) -> List[Dict[str, Any]]:
    """
    Search for semantically similar chunks with balanced representation from different documents.
    
    Implements a hybrid search strategy that combines:
    1. Balanced representation from all available documents
    2. Quality prioritization of most relevant chunks overall
    3. Intelligent scoring with length-based bonuses
    4. Deduplication and final ranking
    
    Args:
        query_embedding (List[float]): Embedding vector of the user's query
        limit (int): Maximum number of chunks to return (default: 12)
        
    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries with content, metadata, and scores
        
    Strategy Overview:
        Phase 1: Scan all chunks from DynamoDB with pagination
        Phase 2: Group chunks by source document for balanced selection
        Phase 3: Calculate similarity scores with length bonuses
        Phase 4: Select top chunks from each document + overall top chunks
        Phase 5: Deduplicate and return final ranked results
        
    """
    # Access DynamoDB table from environment variable
    table = dynamodb.Table(os.environ['EMBEDDINGS_TABLE'])
    
    # Scan all items with pagination to handle large datasets
    items = []
    last_evaluated_key = None
    
    print("Scanning DynamoDB table for chunks...")
    
    # Pagination loop to retrieve all items from DynamoDB
    while True:
        if last_evaluated_key:
            response = table.scan(ExclusiveStartKey=last_evaluated_key)
        else:
            response = table.scan()
        
        items.extend(response.get('Items', []))
        last_evaluated_key = response.get('LastEvaluatedKey')
        
        # Break loop when no more items to scan (no LastEvaluatedKey)
        if not last_evaluated_key:
            break
    
    print(f"Total chunks in database: {len(items)}")
    
    # Group items by source file for balanced selection strategy
    items_by_source = defaultdict(list)
    for item in items:
        source_file = item.get('source_file', 'unknown')
        items_by_source[source_file].append(item)
    
    print(f"Documents found: {list(items_by_source.keys())}")
    
    # Calculate similarity scores for all chunks with enhanced scoring
    all_similarities = []
    for item in items:
        try:
            # Parse stored embedding from JSON string
            chunk_embedding = json.loads(item['embedding'])
            
            # Calculate base cosine similarity (semantic relevance)
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            
            # Bonus for longer chunks (more contextual information)
            # Longer chunks often contain more complete information and context
            content_length = len(item.get('content', ''))
            length_bonus = min(content_length / 800, 0.15)  # Max 15% bonus
            
            # Calculate final score with length bonus
            final_score = similarity + length_bonus
            all_similarities.append((final_score, item))
            
        except Exception as e:
            print(f"Error processing item {item.get('chunk_id', 'unknown')}: {str(e)}")
            continue
    
    # Sort all chunks by descending similarity score
    all_similarities.sort(key=lambda x: x[0], reverse=True)
    
    # Strategy 1: Balanced selection - take top chunks from each document
    # Ensures representation from all available documents
    selected_chunks = []
    chunks_per_document = max(2, limit // len(items_by_source)) if items_by_source else limit
    
    for source_file, source_items in items_by_source.items():
        # Filter similarities for this specific document
        doc_similarities = [(score, item) for score, item in all_similarities 
                           if item.get('source_file') == source_file]
        
        # Sort document chunks by score and take top ones
        doc_similarities.sort(key=lambda x: x[0], reverse=True)
        top_doc_chunks = doc_similarities[:chunks_per_document]
        selected_chunks.extend(top_doc_chunks)
        
        print(f"Document {source_file}: selected {len(top_doc_chunks)} chunks")
    
    # Strategy 2: Quality assurance - add overall top chunks
    # Ensures the most relevant chunks are included regardless of source
    overall_top_chunks = all_similarities[:min(6, limit)]
    selected_chunks.extend(overall_top_chunks)
    
    # Remove duplicates and ensure unique chunks while keeping best scores
    unique_chunks = {}
    for score, item in selected_chunks:
        chunk_id = f"{item.get('document_id')}_{item.get('chunk_id')}"
        if chunk_id not in unique_chunks or score > unique_chunks[chunk_id][0]:
            unique_chunks[chunk_id] = (score, item)
    
    # Convert to sorted list by score (descending order)
    final_chunks = sorted(unique_chunks.values(), key=lambda x: x[0], reverse=True)
    
    # Take top N chunks based on the limit parameter
    result_chunks = [item for score, item in final_chunks[:limit]]
    
    print(f"Selected {len(result_chunks)} chunks from {len(unique_chunks)} unique chunks")
    
    # Log final distribution for debugging and monitoring
    source_count = defaultdict(int)
    for chunk in result_chunks:
        source_file = chunk.get('source_file', 'unknown')
        source_count[source_file] += 1
    
    print("Final chunk distribution:")
    for source, count in source_count.items():
        print(f"  {source}: {count} chunks")
    
    return result_chunks

def invoke_claude_3(prompt: str) -> str:
    """
    Invoke Anthropic Claude 3 model to generate natural language responses.
    
    Supports both Claude 3 Sonnet (default) and Claude 3.5 Haiku models.
    Sonnet provides higher quality responses while Haiku offers better
    cost-performance ratio for simpler queries.
    
    Args:
        prompt (str): Complete prompt with context and question formatted for Claude
        
    Returns:
        str: Generated response text from Claude 3
        
    Model Options:
        - anthropic.claude-3-sonnet-20240229-v1:0: Higher quality, more expensive
        - anthropic.claude-3-5-haiku-v1:0: Faster, more cost-effective
    """
    # Model selection - currently using Sonnet, Haiku is commented out for alternative
    # CLAUDE_MODEL = 'anthropic.claude-3-sonnet-20240229-v1:0'
    CLAUDE_MODEL = 'anthropic.claude-3-5-haiku-20241022-v1:0'  # Alternative: faster, cheaper
    
    try:
        # Invoke Claude 3 model with structured prompt following Anthropic's message format
        # response = bedrock.invoke_model(
        #     modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        #     body=json.dumps({
        #         "anthropic_version": "bedrock-2023-05-31",
        #         "max_tokens": 1500,  # Increased token limit for comprehensive responses
        #         "messages": [{"role": "user", "content": prompt}]
        #     })
        # )
        # Invoke Claude 3.5 Haiku model with structured prompt following Anthropic's message format
        response = bedrock.invoke_model(
            modelId=CLAUDE_MODEL,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,        # Controls randomness (0-1)
                "top_p": 0.9,              # Diversity control
                "top_k": 50,               # Limited token options
            })
        )
        
        # Parse and return the generated text from response
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
        
    except Exception as e:
        print(f"Error invoking Claude 3: {str(e)}")
        # Return error message instead of raising to maintain API response stability
        return f"Error: Unable to generate response. {str(e)}"

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function for processing RAG queries.
    
    Main entry point that handles the complete query processing pipeline:
    - HTTP request parsing from API Gateway
    - Query embedding generation using Titan
    - Semantic search in DynamoDB for relevant chunks
    - Response generation using Claude 3
    - HTTP response formatting with CORS headers
    
    Args:
        event (Dict): Lambda event containing API Gateway request data
        context (Any): Lambda context object with runtime information
        
    Returns:
        Dict: HTTP response with status code, headers, and body
        
    Example Event:
        {
            "queryStringParameters": {"question": "What are the policies?"},
            "requestContext": {"identity": {"sourceIp": "192.168.1.1"}}
        }
        
    Response Format:
        {
            "answer": "Generated response...",
            "sources": ["file1.pdf", "file2.csv"],
            "context_chunks": 5,
            "documents_used": 2
        }
    """
    try:
        # Log incoming event for debugging and monitoring
        print(f"Event received: {json.dumps(event)}")
        
        # Parse question from different possible event formats
        query = ''
        if 'queryStringParameters' in event and event['queryStringParameters']:
            query = event['queryStringParameters'].get('question', '')
        elif 'body' in event and event['body']:
            try:
                body = json.loads(event['body'])
                query = body.get('question', '')
            except json.JSONDecodeError:
                query = ''
        
        # Handle empty query with proper error response
        if not query.strip():
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'No question provided'})
            }
        
        print(f"Processing query: '{query}'")
        
        # Step 1: Generate embedding for the query using Titan
        print("Generating query embedding...")
        query_embedding = get_titan_embeddings(query)
        print(f"Query embedding generated: {len(query_embedding)} dimensions")
        
        # Step 2: Search for similar chunks using balanced approach
        print("Searching for similar chunks...")
        similar_chunks = search_similar_chunks_balanced(query_embedding, limit=40)
        
        # Handle case where no relevant chunks are found
        if not similar_chunks:
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'answer': 'No relevant information found in the documents. '
                             'Please try a different question or ensure documents have been processed.',
                    'sources': [],
                    'context_chunks': 0,
                    'documents_used': 0
                })
            }
        
        # Log search results for monitoring and debugging
        unique_documents = len(set(chunk['source_file'] for chunk in similar_chunks))
        print(f"Found {len(similar_chunks)} relevant chunks from {unique_documents} documents")
        
        # Step 3: Build context from retrieved chunks, grouped by document
        context_parts = []
        current_document = None
        
        for chunk in similar_chunks:
            source_file = chunk['source_file']
            if source_file != current_document:
                context_parts.append(f"\n--- Document: {source_file} ---")
                current_document = source_file
            context_parts.append(chunk['content'])
        
        context = "\n".join(context_parts)
        
        # Step 4: Create comprehensive prompt for Claude 3
        # Prompt engineering to encourage multi-source analysis and citation
        prompt_template = """You are an expert assistant that analyzes information from multiple documents. 

            IMPORTANT INSTRUCTIONS:
            1. Analyze ALL documents provided in the context
            2. Combine relevant information from ALL available sources
            3. If different documents have complementary information, integrate it
            4. If there is contradictory information between documents, mention it
            5. Respond in the same language as the question
            6. Cite specific documents when using information from them
            7. If you cannot find the answer in the context, clearly state this

            CONTEXT FROM MULTIPLE DOCUMENTS:
            {context}

            QUESTION: {question}

            Provide a comprehensive response integrating information from ALL relevant documents. 
            If you use information from a specific document, mention it briefly."""

        prompt = prompt_template.format(context=context, question=query)
        
        # Step 5: Generate response using Claude 3
        print("Generating response with Claude 3...")
        response = invoke_claude_3(prompt)
        
        # Step 6: Prepare response data with metadata
        sources = list(set([chunk['source_file'] for chunk in similar_chunks]))
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'  # CORS enabled for web applications
            },
            'body': json.dumps({
                'answer': response,
                'sources': sources,
                'context_chunks': len(similar_chunks),
                'documents_used': len(sources)
            })
        }
        
    except Exception as e:
        # Comprehensive error handling and logging
        print(f"Error processing query: {str(e)}")
        import traceback
        traceback.print_exc()  # Full traceback for debugging
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error processing your query',
                'details': str(e),
                'suggestion': 'Please try again later or contact support'
            })
        }
