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
from typing import Dict, List, Tuple

# Initialize AWS clients
session = boto3.Session()
s3 = session.client('s3')
dynamodb = session.resource('dynamodb')
bedrock = session.client('bedrock-runtime')

# Configuration constants
DEFAULT_MAX_CHUNKS = 50
MAX_TEXT_LENGTH_FOR_EMBEDDING = 10000
EMBEDDING_MODEL = 'amazon.titan-embed-text-v2:0'

class ChunkingConfig:
    """
    Configuration class for dynamic chunk sizing based on file characteristics
    """
    # Chunk size parameters (in characters)
    CHUNK_SIZES = {
        'PDF': {
            'small': 2000,    # For PDFs < 100KB
            'medium': 2500,   # For PDFs 100KB - 500KB  
            'large': 3000,    # For PDFs 500KB - 1MB
            'xlarge': 3500    # For PDFs > 1MB
        },
        'CSV': {
            'small': 1500,    # For CSVs < 50KB
            'medium': 2000,   # For CSVs 50KB - 200KB
            'large': 2500,    # For CSVs 200KB - 500KB
            'xlarge': 3000    # For CSVs > 500KB
        }
    }
    
    # Chunk overlap percentages
    OVERLAP_PERCENTAGES = {
        'PDF': 0.15,  # 15% overlap for PDFs (better context retention)
        'CSV': 0.10   # 10% overlap for CSVs (less overlap needed for structured data)
    }
    
    # Maximum chunks based on file size (in bytes)
    MAX_CHUNKS_CONFIG = {
        'PDF': {
            'small': 20,     # < 100KB
            'medium': 50,    # 100KB - 500KB
            'large': 80,     # 500KB - 1MB
            'xlarge': 120    # > 1MB
        },
        'CSV': {
            'small': 15,     # < 50KB
            'medium': 30,    # 50KB - 200KB
            'large': 50,     # 200KB - 500KB
            'xlarge': 75     # > 500KB
        }
    }
    
    @staticmethod
    def get_size_category(file_size: int, file_type: str) -> str:
        """
        Determine size category based on file size and type
        
        Args:
            file_size: Size of the file in bytes
            file_type: Type of file ('PDF' or 'CSV')
            
        Returns:
            str: Size category ('small', 'medium', 'large', 'xlarge')
        """
        if file_type == 'PDF':
            if file_size < 100 * 1024:       # < 100KB
                return 'small'
            elif file_size < 500 * 1024:     # 100KB - 500KB
                return 'medium'
            elif file_size < 1024 * 1024:    # 500KB - 1MB
                return 'large'
            else:                            # > 1MB
                return 'xlarge'
        else:  # CSV
            if file_size < 50 * 1024:        # < 50KB
                return 'small'
            elif file_size < 200 * 1024:     # 50KB - 200KB
                return 'medium'
            elif file_size < 500 * 1024:     # 200KB - 500KB
                return 'large'
            else:                            # > 500KB
                return 'xlarge'
    
    @staticmethod
    def get_chunking_parameters(file_size: int, file_type: str) -> Dict:
        """
        Get optimal chunking parameters based on file characteristics
        
        Args:
            file_size: Size of the file in bytes
            file_type: Type of file ('PDF' or 'CSV')
            
        Returns:
            Dict: Chunking parameters including size, overlap, and max chunks
        """
        size_category = ChunkingConfig.get_size_category(file_size, file_type)
        
        return {
            'chunk_size': ChunkingConfig.CHUNK_SIZES[file_type][size_category],
            'chunk_overlap': int(ChunkingConfig.CHUNK_SIZES[file_type][size_category] * 
                                ChunkingConfig.OVERLAP_PERCENTAGES[file_type]),
            'max_chunks': ChunkingConfig.MAX_CHUNKS_CONFIG[file_type][size_category],
            'size_category': size_category
        }

def get_titan_embeddings(text: str) -> List[float]:
    """
    Generate embeddings using Amazon Titan Embeddings model
    
    Args:
        text: Input text to generate embeddings for
        
    Returns:
        List[float]: Embedding vector
        
    Raises:
        Exception: If embedding generation fails
    """
    try:
        # Truncate very long texts to avoid token limits and improve performance
        if len(text) > MAX_TEXT_LENGTH_FOR_EMBEDDING:
            text = text[:MAX_TEXT_LENGTH_FOR_EMBEDDING]
            print(f"Text truncated to {MAX_TEXT_LENGTH_FOR_EMBEDDING} characters for embeddings")
            
        response = bedrock.invoke_model(
            modelId=EMBEDDING_MODEL,
            body=json.dumps({'inputText': text})
        )
        response_body = json.loads(response['body'].read())
        return response_body['embedding']
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        raise

def process_pdf(bucket: str, key: str, file_size: int) -> str:
    """
    Process PDF file and extract text content
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        file_size: Size of the PDF file in bytes
        
    Returns:
        str: Extracted text content
        
    Raises:
        Exception: If PDF processing fails
    """
    try:
        decoded_key = urllib.parse.unquote(key)
        print(f"Processing PDF: {decoded_key}, Size: {file_size} bytes")
        
        start_time = time.time()
        response = s3.get_object(Bucket=bucket, Key=decoded_key)
        pdf_content = response['Body'].read()
        download_time = time.time() - start_time
        
        print(f"PDF downloaded in {download_time:.2f}s, size: {len(pdf_content)} bytes")
        
        # Process PDF
        start_time = time.time()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        num_pages = len(pdf_reader.pages)
        text = ""
        
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"--- Page {i+1} ---\n{page_text}\n\n"
            # Log progress every 10 pages for large documents
            if (i + 1) % 10 == 0 or (i + 1) == num_pages:
                print(f"Processed page {i+1}/{num_pages}")
        
        process_time = time.time() - start_time
        print(f"PDF processed in {process_time:.2f}s, {num_pages} pages, {len(text)} characters")
        
        return text.strip()
    except Exception as e:
        print(f"Error processing PDF {key}: {str(e)}")
        raise

def process_csv(bucket: str, key: str, file_size: int) -> str:
    """
    Process CSV file with multiple encoding attempts
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        file_size: Size of the CSV file in bytes
        
    Returns:
        str: Extracted text content
        
    Raises:
        Exception: If CSV processing fails
    """
    try:
        decoded_key = urllib.parse.unquote(key)
        print(f"Processing CSV: {decoded_key}, Size: {file_size} bytes")
        
        response = s3.get_object(Bucket=bucket, Key=decoded_key)
        csv_content = response['Body'].read()
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                decoded_content = csv_content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            # Fallback to utf-8 with error replacement
            decoded_content = csv_content.decode('utf-8', errors='replace')
        
        csv_reader = csv.reader(io.StringIO(decoded_content))
        text = ""
        
        for i, row in enumerate(csv_reader):
            if i >= 1000:  # Safety limit for very large CSVs
                text += f"... (truncated after 1000 rows)\n"
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

def split_text(text: str, chunk_size: int, chunk_overlap: int, max_chunks: int) -> List[str]:
    """
    Split text into chunks with optimal parameters
    
    Args:
        text: Text to split into chunks
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        max_chunks: Maximum number of chunks to generate
        
    Returns:
        List[str]: List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text) and len(chunks) < max_chunks:
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move to next chunk position with overlap
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    
    print(f"Split into {len(chunks)} chunks (max allowed: {max_chunks})")
    print(f"Chunk size: {chunk_size} chars, Overlap: {chunk_overlap} chars")
    
    return chunks

def lambda_handler(event, context):
    """
    AWS Lambda handler for processing uploaded documents
    
    Args:
        event: AWS Lambda event containing S3 trigger information
        context: AWS Lambda context object
        
    Returns:
        Dict: Processing results with status code and message
    """
    try:
        print(f"Event received: {json.dumps(event)}")
        
        # Get bucket and key from S3 event
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        print(f"Processing file: s3://{bucket}/{key}")
        
        # Get file metadata including size
        file_info = s3.head_object(Bucket=bucket, Key=urllib.parse.unquote(key))
        file_size = file_info['ContentLength']
        print(f"File size: {file_size} bytes")
        
        # Determine file type and chunking parameters
        decoded_key = urllib.parse.unquote(key)
        if decoded_key.lower().endswith('.pdf'):
            file_type = 'PDF'
            text = process_pdf(bucket, decoded_key, file_size)
        elif decoded_key.lower().endswith('.csv'):
            file_type = 'CSV'
            text = process_csv(bucket, decoded_key, file_size)
        else:
            print(f"Unsupported file format: {decoded_key}")
            return {
                'statusCode': 400,
                'body': json.dumps('Unsupported file format. Only PDF and CSV are supported.')
            }
        
        print(f"Extracted text length: {len(text)} characters")
        print(f"File type: {file_type}")
        
        # Get dynamic chunking parameters based on file characteristics
        chunking_params = ChunkingConfig.get_chunking_parameters(file_size, file_type)
        print(f"Chunking parameters: {chunking_params}")
        
        # Split text into chunks using dynamic parameters
        chunks = split_text(
            text, 
            chunking_params['chunk_size'], 
            chunking_params['chunk_overlap'], 
            chunking_params['max_chunks']
        )
        
        # Store chunks and embeddings in DynamoDB
        table = dynamodb.Table(os.environ['EMBEDDINGS_TABLE'])
        document_id = str(uuid.uuid4())
        
        processed_chunks = 0
        failed_chunks = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Skip empty or very short chunks
                if not chunk.strip() or len(chunk.strip()) < 50:
                    print(f"Skipping empty/short chunk {i}")
                    continue
                    
                print(f"Processing chunk {i+1}/{len(chunks)}: {len(chunk)} characters")
                
                # Generate embedding with retry logic
                max_retries = 3
                embedding = None
                
                for attempt in range(max_retries):
                    try:
                        embedding = get_titan_embeddings(chunk)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        print(f"Retry {attempt + 1} for chunk {i} failed: {str(e)}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                
                # Store in DynamoDB with optimized item structure
                item = {
                    'document_id': document_id,
                    'chunk_id': f'chunk_{i}',
                    'content': chunk[:2000],  # Limit content size for DynamoDB
                    'embedding': json.dumps(embedding),
                    'source_file': decoded_key,
                    'file_type': file_type,
                    'file_size': file_size,
                    'size_category': chunking_params['size_category'],
                    'chunk_size': len(chunk),
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'created_at': datetime.utcnow().isoformat()
                }
                
                table.put_item(Item=item)
                processed_chunks += 1
                print(f"Successfully stored chunk {i}")
                
            except Exception as e:
                failed_chunks += 1
                print(f"Error processing chunk {i}: {str(e)}")
                continue
        
        total_time = context.get_remaining_time_in_millis() / 1000.0
        print(f"Processing completed in {total_time:.2f} seconds")
        print(f"Successfully processed {processed_chunks} chunks, failed: {failed_chunks}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Processed {processed_chunks} chunks from {decoded_key}',
                'document_id': document_id,
                'chunks_processed': processed_chunks,
                'chunks_failed': failed_chunks,
                'file_type': file_type,
                'file_size': file_size,
                'size_category': chunking_params['size_category'],
                'chunk_size': chunking_params['chunk_size'],
                'chunk_overlap': chunking_params['chunk_overlap'],
                'max_chunks': chunking_params['max_chunks'],
                'processing_time': total_time
            })
        }
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error processing file: {str(e)}')
        }