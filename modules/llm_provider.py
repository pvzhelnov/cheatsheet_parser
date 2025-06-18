"""
LLM Provider SDK - Unified interface for multiple LLM providers
"""

import os
import json
import yaml
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Type, Union, List
from abc import ABC, abstractmethod, ABCMeta
from pydantic import BaseModel

import requests
from google import genai
import ollama
import base64
import pymupdf
from openai import OpenAI

from google.genai import types

class NotReadyMeta(ABCMeta):
    def __new__(mcs, name, bases, dct):
        for attr_name, attr_value in dct.items():
            mask = ((not attr_name.startswith('__')) or (attr_name == '__init__'))
            if callable(attr_value) and mask:
                dct[attr_name] = NotReadyMeta.raise_not_implemented_wrapper(attr_value)
        return super().__new__(mcs, name, bases, dct)

    @staticmethod
    def raise_not_implemented_wrapper(func):
        def wrapper(*args, **kwargs):
            raise NotImplementedError(f"Class '{args[0].__class__.__name__}' is not ready for use yet. Method '{func.__name__}' is not implemented.")
        return wrapper
    
class BasePrompt(BaseModel):
    prompt: Optional[Union[str, List[str]]] = None
    hash: Optional[str] = None

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs:
            kwargs = {'prompt': args[0]}
        elif len(args) > 1:
            raise TypeError("Only one positional argument allowed.")
        super().__init__(**kwargs)
        self.hash = self._version_prompt()

    def _version_prompt(self) -> str | None:
        """Version a prompt and save it"""

        if self.prompt is None:
            return

        # Stringify if List[str] before hashing
        stringified_prompt = str(self.prompt)

        prompt_hash = hashlib.sha256(stringified_prompt.encode()).hexdigest()[:8]
        
        prompts_dir = Path("prompts")
        prompts_dir.mkdir(exist_ok=True)
        
        prompt_file = prompts_dir / f"{prompt_hash}.txt"
        if not prompt_file.exists():
            prompt_file.write_text(stringified_prompt)
            
        return prompt_hash
    
    def __str__(self):
        return str(self.prompt or "")
    
    def parts(self):
        if isinstance(self.prompt, list):
            prompt_parts = self.prompt
        else:
            prompt_parts = [str(self.prompt)] if self.prompt else []
        return prompt_parts

class LLMResponse(BaseModel):
    """Standard response format for all LLM providers"""
    content: BaseModel = None
    token_usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = {}

class BaseResponseSchema(BaseModel):
    response: str

class BaseLLM(BaseModel):
    """Base class for all LLMs"""
    model_name: Optional[str] = None
    response_schema: Type[BaseModel] = BaseResponseSchema
    system_instruction: BasePrompt = BasePrompt()
    temperature: Optional[float] = 0.0
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.95
    seed: Optional[int] = 42
    safety_settings: Optional[Any] = None

class GeminiLLM(BaseLLM):
    model_name: Optional[str] = "gemini-2.0-flash"

class BaseLLMProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, **kwargs):
        self.provider_name = self.__class__.__name__.lower().replace('provider', '')
        llm_type = kwargs.pop('model', BaseLLM)
        safety_settings = kwargs.pop('safety_settings', self.unsafe_settings())
        system_instruction = BasePrompt(kwargs.pop('system_instruction', None))
        self.model: BaseLLM = llm_type(
            safety_settings=safety_settings,
            system_instruction=system_instruction,
            **kwargs
        )

    @abstractmethod
    def unsafe_settings(self):
        """Return provider-specific full uncensored safety settings"""
        pass
        
    @abstractmethod
    def generate(self, prompt: BasePrompt, response_schema: Optional[Type[BaseModel]] = None) -> LLMResponse:
        """Generate response from the LLM"""
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, **kwargs):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        self.client = genai.Client(api_key=api_key)
        kwargs.setdefault('model', GeminiLLM)
        super().__init__(**kwargs)

    def unsafe_settings(self):
        # https://ai.google.dev/api/generate-content#v1beta.HarmCategory
        return [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            )
        ]
        
    def generate(self, prompt: BasePrompt) -> LLMResponse:
        try:                
            contents = []

            # Process each item in the prompt
            for item in prompt.parts():
                if self._is_valid_path_or_url(item):
                    uploaded_file = self._upload_file(item)
                    contents.append(uploaded_file)
                else:
                    contents.append(item)
            
            # Add schema instruction as separate item if response_schema is specified
            response_schema = self.model.response_schema
            if response_schema:
                schema_instruction = f"Respond with valid JSON matching this schema: {response_schema.model_json_schema()}"
                contents.append(schema_instruction)
                
            request_timestamp = datetime.now()
            response = self.client.models.generate_content(
                contents=contents,
                model=self.model.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=str(self.model.system_instruction),
                    temperature=self.model.temperature,
                    top_k=self.model.top_k,
                    top_p=self.model.top_p,
                    seed=self.model.seed,
                    safety_settings=self.model.safety_settings,
                    response_mime_type='application/json',
                    response_schema=response_schema
                    #response_json_schema=response_schema.model_json_schema()  # not implemented
                )
            )
            
            return LLMResponse(
                content=response_schema.model_validate_json(response.text),
                token_usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                    "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                } if hasattr(response, 'usage_metadata') else None,
                metadata={
                    "request_timestamp": request_timestamp.isoformat(),
                    "response_timestamp": datetime.now().isoformat(),
                    "finish_reason": "completed"
                }
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")
        
    def _is_valid_path_or_url(self, prompt_part: str) -> bool:
        """Check if prompt is a valid local path or accessible URL"""
        # Check if it's a local file path
        if Path(prompt_part.strip()).exists():
            return True
        
        # Check if it's a URL
        if prompt_part.strip().startswith(('http://', 'https://')):
            try:
                response = requests.head(prompt_part.strip(), timeout=5)
                return response.status_code == 200
            except:
                return False
        
        return False

    def _upload_file(self, path_or_url: str):
        """Upload file to Gemini"""
        if path_or_url.startswith(('http://', 'https://')):
            # For URLs, download first then upload
            response = requests.get(path_or_url)
            temp_file = f"/tmp/{Path(path_or_url).name}"
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            uploaded_file = self.client.files.upload(file=temp_file)
            os.remove(temp_file)
            return uploaded_file
        else:
            # For local files
            return self.client.files.upload(file=path_or_url.strip())


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider"""
    
    def __init__(self, **kwargs):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        kwargs.setdefault('model', BaseLLM)
        super().__init__(**kwargs)

    def unsafe_settings(self):
        """Return provider-specific full uncensored safety settings"""
        return {}  # OpenRouter doesn't have safety settings like Gemini
        
    def generate(self, prompt: BasePrompt) -> LLMResponse:
        try:
            message_content = []

            # Add text prompts first (recommended by OpenRouter)
            text_parts = []
            file_parts = []

            for item in prompt.parts():
                if self._is_valid_path_or_url(item):
                    file_parts.append(item)
                else:
                    text_parts.append(item)

            # Add text content first
            for text in text_parts:
                message_content.append({"type": "text", "text": text})

            # Then add file content
            for file_path in file_parts:
                file_content = self._process_file(file_path)
                message_content.append(file_content)

            messages = [{"role": "user", "content": message_content}]
            
            # Add schema instruction if response_schema is specified
            response_schema = self.model.response_schema
            if response_schema:
                schema_instruction = f"Respond with valid JSON matching this schema: {response_schema.model_json_schema()}"
                messages.append({"role": "system", "content": schema_instruction})
                
            request_timestamp = datetime.now()
            response = self.client.chat.completions.create(
                model=self.model.model_name,
                messages=messages,
                temperature=self.model.temperature,
                seed=self.model.seed,
                response_format={
                    "type": "json_schema",
                    "json_schema": response_schema.model_json_schema()
                }
            )
            print(messages)
            return LLMResponse(
                content=response.choices[0].message.content,
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None,
                metadata={
                    "request_timestamp": request_timestamp.isoformat(),
                    "response_timestamp": datetime.now().isoformat(),
                    "finish_reason": response.choices[0].finish_reason
                }
            )
        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {str(e)}")
        
    def _is_valid_path_or_url(self, prompt_part: str) -> bool:
        """Check if prompt is a valid local path or accessible URL"""
        # Check if it's a local file path
        if Path(prompt_part.strip()).exists():
            return True
        
        # Check if it's a URL
        if prompt_part.strip().startswith(('http://', 'https://')):
            try:
                response = requests.head(prompt_part.strip(), timeout=5)
                return response.status_code == 200
            except:
                return False
        
        return False

    def _process_file(self, path_or_url: str) -> dict:
        """Process file for OpenRouter - convert to file content type"""
        if path_or_url.startswith(('http://', 'https://')):
            # For URLs, download first
            response = requests.get(path_or_url)
            temp_file = f"/tmp/{Path(path_or_url).name}"
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            file_path = temp_file
        else:
            file_path = path_or_url.strip()
        
        try:
            # Read file as base64
            with open(file_path, 'rb') as f:
                file_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Get filename and MIME type
            filename = Path(file_path).name
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                mime_type = "application/pdf"
                content_part = {
                    "type": "file",
                    "file": {
                        "filename": filename,
                        "file_data": f"data:{mime_type};base64,{file_data}"
                    }
                }
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                # For images, use image_url type
                if file_extension in ['.jpg', '.jpeg']:
                    mime_type = "image/jpeg"
                elif file_extension == '.png':
                    mime_type = "image/png"
                elif file_extension == '.gif':
                    mime_type = "image/gif"
                else:
                    mime_type = "image/png"
                
                content_part = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{file_data}"
                    }
                }
            else:
                # For other file types, use file type
                content_part = {
                    "type": "file",
                    "file": {
                        "filename": filename,
                        "file_data": f"data:application/octet-stream;base64,{file_data}"
                    }
                }
            
        except Exception as e:
            # Fallback: return error as text
            content_part = {"type": "text", "text": f"Error processing file {file_path}: {str(e)}"}
            
        # Clean up temp file if it was downloaded
        if path_or_url.startswith(('http://', 'https://')):
            os.remove(temp_file)
        
        return content_part


class OllamaProvider(BaseLLMProvider):
    """Ollama local provider"""
    
    def __init__(self, **kwargs):
        host = os.getenv("OLLAMA_HOST")
        if not host:
            raise ValueError("OLLAMA_HOST environment variable is required")
        ollama.Client(host=host)
        kwargs.setdefault('model', BaseLLM)
        super().__init__(**kwargs)

    def unsafe_settings(self):
        """Return provider-specific full uncensored safety settings"""
        return {}  # Ollama doesn't have safety settings like Gemini
        
    def generate(self, prompt: BasePrompt) -> LLMResponse:
        try:
            full_prompt = ""
            
            all_images = []
            text_parts = []

            # Process each item in the prompt
            for item in prompt.parts():
                if self._is_valid_path_or_url(item):
                    processed_images = self._process_file(item)
                    all_images.extend(processed_images)
                else:
                    text_parts.append(item)

            full_prompt = "\n".join(text_parts)
            
            # Add schema instruction if response_schema is specified
            response_schema = self.model.response_schema
            if response_schema:
                schema_instruction = f"Respond with valid JSON matching this schema: {response_schema.model_json_schema()}"
                full_prompt += schema_instruction
                
            request_timestamp = datetime.now()
            response = ollama.generate(
                model=self.model.model_name,
                prompt=full_prompt.strip(),
                images=all_images if all_images else None,
                format='json' if response_schema else None,
                options={
                    'temperature': self.model.temperature,
                    'top_k': self.model.top_k,
                    'top_p': self.model.top_p,
                    'seed': self.model.seed
                }
            )
            
            return LLMResponse(
                content=response_schema.model_validate_json(response['response']),
                token_usage={
                    "prompt_tokens": response.get('prompt_eval_count', 0),
                    "completion_tokens": response.get('eval_count', 0),
                    "total_tokens": response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
                },
                metadata={
                    "request_timestamp": request_timestamp.isoformat(),
                    "response_timestamp": datetime.now().isoformat(),
                    "total_duration": response.get('total_duration'),
                    "load_duration": response.get('load_duration'),
                    "prompt_eval_duration": response.get('prompt_eval_duration'),
                    "eval_duration": response.get('eval_duration'),
                    "finish_reason": "completed"
                }
            )
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {str(e)}")
        
    def _is_valid_path_or_url(self, prompt_part: str) -> bool:
        """Check if prompt is a valid local path or accessible URL"""
        # Check if it's a local file path
        if Path(prompt_part.strip()).exists():
            return True
        
        # Check if it's a URL
        if prompt_part.strip().startswith(('http://', 'https://')):
            try:
                response = requests.head(prompt_part.strip(), timeout=5)
                return response.status_code == 200
            except:
                return False
        
        return False

    def _process_file(self, path_or_url: str) -> dict:
        """Process file for Ollama - convert PDF pages to images"""
        if path_or_url.startswith(('http://', 'https://')):
            # For URLs, download first
            response = requests.get(path_or_url)
            temp_file = f"/tmp/{Path(path_or_url).name}"
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            file_path = temp_file
        else:
            file_path = path_or_url.strip()
        
        file_extension = Path(file_path).suffix.lower()
        images = []
        
        try:
            if file_extension == '.pdf':
                # Convert PDF pages to images
                doc = pymupdf.open(file_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x zoom for better quality
                    img_data = pix.tobytes("png")
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    images.append(img_base64)
                doc.close()
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                # For regular images
                with open(file_path, 'rb') as f:
                    file_data = base64.b64encode(f.read()).decode('utf-8')
                images.append(file_data)
        except Exception as e:
            # Fallback: return error as text
            images = []
            
        # Clean up temp file if it was downloaded
        if path_or_url.startswith(('http://', 'https://')):
            os.remove(temp_file)
        
        return images


class LLMAgent:
    """Main agent class for LLM interactions"""
    
    def __init__(self, provider_name: str, **kwargs):
        self.logger = self._setup_logger()
        
        # Provider mapping
        providers: dict[str, BaseLLMProvider] = {
            "gemini": GeminiProvider,
            "openrouter": OpenRouterProvider,
            "ollama": OllamaProvider
        }
        
        if provider_name not in providers:
            raise ValueError(f"Unsupported provider: {provider_name}. Supported: {list(providers.keys())}")
            
        provider_class = providers[provider_name]
        self.provider: BaseLLMProvider = provider_class(**kwargs)
            
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with date-based file structure"""
        now = datetime.now()
        log_dir = Path("logs") / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the calling script name
        import inspect
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back.f_back
            script_name = Path(caller_frame.f_globals.get('__file__', 'unknown')).stem
        finally:
            del frame
            
        log_file = log_dir / f"{script_name}.log"
        
        logger = logging.getLogger(f"llm_agent_{script_name}")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def __call__(self, prompt: Union[str, List[str]], save_response: bool = True) -> LLMResponse:
        """Generate response from LLM"""
        try:
            hashed_prompt = BasePrompt(prompt)

            # Generate response
            response = self.provider.generate(hashed_prompt)

            json_response = response.content.model_dump_json()
            
            # Log the interaction
            log_data = {
                "provider": {
                    "provider_name": self.provider.provider_name,
                    "safety_settings": str(self.provider.model.safety_settings)
                },
                "model": {
                    "model_name": self.provider.model.model_name,
                    "config": {
                        "temperature": self.provider.model.temperature,
                        "top_k": self.provider.model.top_k,
                        "top_p": self.provider.model.top_p,
                        "seed": self.provider.model.seed,
                    }
                },
                "request": {
                    "prompt_hash": hashed_prompt.hash,
                    "system_instruction_hash": self.provider.model.system_instruction.hash,
                    "response_schema": self.provider.model.response_schema.__name__,  # perhaps to be replaced with a hash later on
                },
                "response": {
                    "content": json_response[:200] + "..." if len(json_response) > 200 else json_response,
                    "token_usage": response.token_usage,
                    "metadata": response.metadata,
                }
            }
            self.logger.info(f"LLM Response: {json.dumps(log_data, indent=2, ensure_ascii=False)}")
            
            # Save response if requested
            if save_response:
                self._save_response(response, hashed_prompt)
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise
            
    def _version_prompt(self, prompt: Optional[Union[str, List[str]]]) -> str:
        """Version a prompt and save it"""

        if prompt is None:
            return

        # Stringify if List[str] before hashing
        stringified_prompt = str(prompt)

        prompt_hash = hashlib.sha256(stringified_prompt.encode()).hexdigest()[:8]
        
        prompts_dir = Path("prompts")
        prompts_dir.mkdir(exist_ok=True)
        
        prompt_file = prompts_dir / f"{prompt_hash}.txt"
        if not prompt_file.exists():
            prompt_file.write_text(stringified_prompt)
            
        return prompt_hash
        
    def _save_response(self, response: LLMResponse, prompt: BasePrompt):
        """Save response as YAML file"""
        responses_dir = Path("responses")
        responses_dir.mkdir(exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        response_file = responses_dir / f"{timestamp_str}_{prompt.hash}_{self.provider.provider_name}.yml"
        
        response_data = {
            "provider": {
                "provider_name": self.provider.provider_name,
                "safety_settings": str(self.provider.model.safety_settings)
            },
            "model": {
                "model_name": self.provider.model.model_name,
                "config": {
                    "temperature": self.provider.model.temperature,
                    "top_k": self.provider.model.top_k,
                    "top_p": self.provider.model.top_p,
                    "seed": self.provider.model.seed,
                }
            },
            "request": {
                "prompt_hash": prompt.hash,
                "prompt_file": f"prompts/{prompt.hash}.txt",
                "system_instruction_hash": self.provider.model.system_instruction.hash,
                "system_instruction_file": f"prompts/{self.provider.model.system_instruction.hash}.txt",
                "response_schema": self.provider.model.response_schema.model_json_schema()
            },
            "response": {
                "content": response.content.model_dump(mode='json'),  # to json - dict is hard to load for evals!
                "token_usage": response.token_usage,
                "metadata": response.metadata,
            },
            "evals": {},
            "manual_evals": {},
            "ground_truth": None,
            "name": None
        }
        
        with open(response_file, 'w') as f:
            yaml.dump(response_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def agent(provider: str, response_schema: Type[BaseModel], **kwargs) -> LLMAgent:
    """Factory function to create LLM agent"""
    return LLMAgent(provider_name=provider, response_schema=response_schema, **kwargs)