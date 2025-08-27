import boto3
import json
import uuid
from datetime import datetime
import PyPDF2
import csv
import io
import os
import urllib.parse
import time

# Initialize clients
session = boto3.Session()
s3 = session.client('s3')
dynamodb = session.resource('dynamodb')
bedrock = session.client('bedrock-runtime')

# Configuración
MAX_CHUNKS = int(os.environ.get('MAX_CHUNKS_PER_FILE', 50))
MAX_TEXT_LENGTH = 8000

def get_titan_embeddings(text):
    """Generate embeddings using Amazon Titan"""
    try:
        # Truncate very long texts to avoid token limits
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
            print(f"Text truncated to {MAX_TEXT_LENGTH} characters")
            
        print(f"Generating embeddings for text of length: {len(text)}")
        
        response = bedrock.invoke_model(
            modelId='amazon.titan-embed-text-v2:0',
            body=json.dumps({'inputText': text})
        )
        response_body = json.loads(response['body'].read())
        print("Embeddings generated successfully")
        return response_body['embedding']
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        raise

def process_pdf(bucket, key):
    """Process PDF file"""
    try:
        # Decodificar el key en caso de que esté URL encoded
        decoded_key = urllib.parse.unquote(key)
        print(f"Processing PDF: {decoded_key}")
        
        # Medir tiempo de descarga
        start_time = time.time()
        response = s3.get_object(Bucket=bucket, Key=decoded_key)
        pdf_content = response['Body'].read()
        download_time = time.time() - start_time
        print(f"PDF downloaded in {download_time:.2f} seconds, size: {len(pdf_content)} bytes")
        
        # Procesar PDF
        start_time = time.time()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        num_pages = len(pdf_reader.pages)
        text = ""
        
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"--- Page {i+1} ---\n{page_text}\n\n"
            print(f"Processed page {i+1}/{num_pages}")
        
        process_time = time.time() - start_time
        print(f"PDF processed in {process_time:.2f} seconds, {num_pages} pages, {len(text)} characters")
        
        return text.strip()
    except Exception as e:
        print(f"Error processing PDF {key}: {str(e)}")
        raise

def process_csv(bucket, key):
    """Process CSV file with multiple encoding attempts"""
    try:
        # Decodificar el key en caso de que esté URL encoded
        decoded_key = urllib.parse.unquote(key)
        print(f"Processing CSV: {decoded_key}")
        
        response = s3.get_object(Bucket=bucket, Key=decoded_key)
        csv_content = response['Body'].read()
        
        # Intentar diferentes codificaciones
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                decoded_content = csv_content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            # Si ninguna codificación funciona, usar utf-8 y reemplazar caracteres problemáticos
            decoded_content = csv_content.decode('utf-8', errors='replace')
        
        csv_reader = csv.reader(io.StringIO(decoded_content))
        text = ""
        
        for i, row in enumerate(csv_reader):
            if i >= 100:  # Limitar a 100 filas máximo
                text += f"... (truncated after 100 rows)\n"
                break
                
            if i == 0:  # Header row
                text += "Headers: " + " | ".join(row) + "\n"
            else:
                text += "Row " + str(i) + ": " + " | ".join(row) + "\n"
        
        print(f"CSV processed: {i+1} rows, {len(text)} characters")
        return text.strip()
    except Exception as e:
        print(f"Error processing CSV {key}: {str(e)}")
        raise

def split_text(text, chunk_size=1500, chunk_overlap=200):
    """Split text into chunks"""
    chunks = []
    start = 0
    
    while start < len(text) and len(chunks) < MAX_CHUNKS:
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    
    print(f"Split into {len(chunks)} chunks")
    return chunks

def lambda_handler(event, context):
    try:
        print(f"Event received: {json.dumps(event)}")
        start_time = time.time()
        
        # Get bucket and key from S3 event
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        print(f"Processing file: s3://{bucket}/{key}")
        
        # Decodificar el key para logging
        decoded_key = urllib.parse.unquote(key)
        print(f"Decoded key: {decoded_key}")
        
        # Determine file type
        if decoded_key.lower().endswith('.pdf'):
            text = process_pdf(bucket, decoded_key)
        elif decoded_key.lower().endswith('.csv'):
            text = process_csv(bucket, decoded_key)
        else:
            print(f"Unsupported file format: {decoded_key}")
            return {
                'statusCode': 400,
                'body': json.dumps('Unsupported file format. Only PDF and CSV are supported.')
            }
        
        print(f"Extracted text length: {len(text)} characters")
        
        # Split text into chunks
        chunks = split_text(text)
        print(f"Split into {len(chunks)} chunks")
        
        # Store chunks and embeddings in DynamoDB
        table = dynamodb.Table(os.environ['EMBEDDINGS_TABLE'])
        document_id = str(uuid.uuid4())
        
        processed_chunks = 0
        failed_chunks = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Skip empty chunks
                if not chunk.strip():
                    print(f"Skipping empty chunk {i}")
                    continue
                
                print(f"Processing chunk {i+1}/{len(chunks)}: {len(chunk)} characters")
                
                # Generate embedding with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        embedding = get_titan_embeddings(chunk)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        print(f"Retry {attempt + 1} for chunk {i} failed: {str(e)}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                
                # Store in DynamoDB
                item = {
                    'document_id': document_id,
                    'chunk_id': f'chunk_{i}',
                    'content': chunk[:2000],  # Limitar tamaño para DynamoDB
                    'embedding': json.dumps(embedding),
                    'source_file': decoded_key,
                    'file_type': 'PDF' if decoded_key.lower().endswith('.pdf') else 'CSV',
                    'created_at': datetime.utcnow().isoformat(),
                    'chunk_size': len(chunk),
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                
                table.put_item(Item=item)
                processed_chunks += 1
                print(f"Successfully stored chunk {i}")
                
            except Exception as e:
                failed_chunks += 1
                print(f"Error processing chunk {i}: {str(e)}")
                continue
        
        total_time = time.time() - start_time
        print(f"Processing completed in {total_time:.2f} seconds")
        print(f"Successfully processed {processed_chunks} chunks, failed: {failed_chunks}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Processed {processed_chunks} chunks from {decoded_key}',
                'document_id': document_id,
                'chunks_processed': processed_chunks,
                'chunks_failed': failed_chunks,
                'processing_time': total_time
            })
        }
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error processing file: {str(e)}')
        }