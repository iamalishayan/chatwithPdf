import json
import time
import asyncio
import logging
from typing import List, Dict, Any, AsyncGenerator
import httpx
from fastapi import HTTPException
from app.config import settings

logger = logging.getLogger("gemini-client")

async def get_embedding(text: str) -> List[float]:
    """Fetches text embedding using gemini-embedding-001 from Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={settings.GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "content": {
            "parts": [{"text": text}]
        }
    }
    
    async with httpx.AsyncClient() as client:
        for attempt in range(3):
            try:
                response = await client.post(url, headers=headers, json=payload, timeout=15.0)
                if response.status_code == 200:
                    data = response.json()
                    return data["embedding"]["values"]
                else:
                    logger.warning(f"Embedding API error (Status {response.status_code}): {response.text}")
            except Exception as e:
                logger.warning(f"Embedding request failed (attempt {attempt+1}): {e}")
            await asyncio.sleep(1)
        
        raise HTTPException(status_code=502, detail="Failed to retrieve embeddings from Gemini API.")

async def stream_answer(
    context: str,
    question: str,
    sources: List[Dict[str, Any]]
) -> AsyncGenerator[str, None]:
    """Streams the response from Gemini 2.5 Flash via SSE, including source metadata and TTFT metrics."""
    # 1. Send sources immediately
    yield f"data: {json.dumps({'event': 'sources', 'data': sources})}\n\n"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key={settings.GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    system_prompt = (
        "You are an expert assistant designed to answer questions using the provided document context.\n"
        "Strictly answer the question based ONLY on the context below. If the context does not contain the answer, "
        "say 'I cannot find the answer in the provided document.' and do not make up information.\n\n"
        f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---"
    )
    
    payload = {
        "contents": [
            {
                "parts": [{"text": question}]
            }
        ],
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048
        }
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    error_detail = await response.aread()
                    logger.error(f"Gemini API returned error code {response.status_code}: {error_detail}")
                    yield f"data: {json.dumps({'event': 'error', 'error': f'Gemini API error status code {response.status_code}'})}\n\n"
                    return
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        try:
                            chunk_data = json.loads(data_str)
                            candidates = chunk_data.get("candidates", [])
                            if candidates:
                                text_part = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                                if text_part:
                                    if first_token_time is None:
                                        first_token_time = time.perf_counter()
                                    
                                    yield f"data: {json.dumps({'event': 'token', 'text': text_part})}\n\n"
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode JSON chunk: {data_str}")
            
            # Send performance metrics at the end of successful streaming
            end_time = time.perf_counter()
            total_duration = end_time - start_time
            ttft = (first_token_time - start_time) if first_token_time else None
            
            metrics_payload = {
                'event': 'metrics',
                'ttft_seconds': round(ttft, 3) if ttft is not None else None,
                'total_duration_seconds': round(total_duration, 3)
            }
            yield f"data: {json.dumps(metrics_payload)}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: {json.dumps({'event': 'error', 'error': str(e)})}\n\n"
