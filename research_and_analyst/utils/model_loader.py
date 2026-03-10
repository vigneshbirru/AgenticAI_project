import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from research_and_analyst.utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from research_and_analyst.logger import GLOBAL_LOGGER as log
from research_and_analyst.exception.custom_exception import ResearchAnalystException


class ApiKeyManager:
    """
    Loads and manages all environment-based API keys.
    """

    def __init__(self):
        load_dotenv()

        self.api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        }

        log.info("Initializing ApiKeyManager")

        # Log loaded key statuses without exposing secrets
        for key, val in self.api_keys.items():
            if val:
                log.info(f"{key} loaded successfully from environment")
            else:
                log.warning(f"{key} is missing in environment variables")

    def get(self, key: str):
        """
        Retrieve a specific API key.

        Args:
            key (str): Name of the API key.

        Returns:
            str | None: API key value if found.
        """
        return self.api_keys.get(key)


class ModelLoader:
    """
    Loads embedding models and LLMs dynamically based on YAML configuration and environment settings.
    """

    def __init__(self):
        """
        Initialize the ModelLoader and load configuration.
        """
        try:
            self.api_key_mgr = ApiKeyManager()
            self.config = load_config()
            log.info("YAML configuration loaded successfully", config_keys=list(self.config.keys()))
        except Exception as e:
            log.error("Error initializing ModelLoader", error=str(e))
            raise ResearchAnalystException("Failed to initialize ModelLoader", sys)

    # ----------------------------------------------------------------------
    # Embedding Loader
    # ----------------------------------------------------------------------
    def load_embeddings(self):
        """
        Load and return a Google Generative AI embedding model.

        Returns:
            GoogleGenerativeAIEmbeddings: Loaded embedding model instance.
        """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)

            # Ensure event loop exists for gRPC-based embedding API
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
            )

            log.info("Embedding model loaded successfully", model=model_name)
            return embeddings

        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise ResearchAnalystException("Failed to load embedding model", sys)

    # ----------------------------------------------------------------------
    # LLM Loader
    # ----------------------------------------------------------------------
    def load_llm(self):
        """
        Load and return a chat-based LLM according to the configured provider.

        Supported providers:
            - OpenAI
            - Google (Gemini)
            - Groq

        Returns:
            ChatOpenAI | ChatGoogleGenerativeAI | ChatGroq: LLM instance
        """
        try:
            llm_block = self.config["llm"]
            provider_key = os.getenv("LLM_PROVIDER")
            if provider_key:
                provider_key = provider_key.strip().lower()

            key_by_provider = {
                "openai": "OPENAI_API_KEY",
                "google": "GOOGLE_API_KEY",
                "groq": "GROQ_API_KEY",
            }

            # If provider isn't explicitly chosen, prefer whatever has a key present.
            if not provider_key:
                preferred = []
                if self.api_key_mgr.get("GROQ_API_KEY"):
                    preferred.append("groq")
                if self.api_key_mgr.get("GOOGLE_API_KEY"):
                    preferred.append("google")
                if self.api_key_mgr.get("OPENAI_API_KEY"):
                    preferred.append("openai")
                provider_key = next((k for k in preferred if k in llm_block), "openai")

            # If a provider was requested but its API key is missing, fall back to an
            # available provider (instead of failing with an OpenAI/Google SDK error).
            if provider_key in llm_block:
                requested_provider = (llm_block[provider_key].get("provider") or provider_key).strip().lower()
                required_key = key_by_provider.get(requested_provider)
                if required_key and not self.api_key_mgr.get(required_key):
                    log.warning(
                        "Requested LLM provider is missing its API key; falling back to available provider",
                        requested_provider=requested_provider,
                        missing_key=required_key,
                    )
                    provider_key = next(
                        (
                            k
                            for k in ["groq", "google", "openai"]
                            if k in llm_block and self.api_key_mgr.get(key_by_provider.get(k, ""))
                        ),
                        provider_key,
                    )

            if provider_key not in llm_block:
                log.error("LLM provider not found in configuration", provider=provider_key)
                raise ValueError(f"LLM provider '{provider_key}' not found in configuration")

            llm_config = llm_block[provider_key]
            provider = (llm_config.get("provider") or provider_key).strip().lower()
            model_name = llm_config.get("model_name")
            temperature = llm_config.get("temperature", 0.2)
            max_tokens = llm_config.get("max_output_tokens", 2048)

            # Allow model override without editing YAML.
            # Examples (PowerShell):
            #   $env:LLM_MODEL_NAME="..."
            #   $env:GROQ_MODEL_NAME="..."
            env_model_name = os.getenv("LLM_MODEL_NAME")
            env_model_name = os.getenv(f"{provider_key.upper()}_MODEL_NAME") or env_model_name
            env_model_name = os.getenv(f"{provider.upper()}_MODEL_NAME") or env_model_name
            if env_model_name:
                model_name = env_model_name
                log.info("Overriding LLM model from environment", provider=provider, model=model_name)

            log.info("Loading LLM", provider=provider, model=model_name)

            if provider == "google":
                if not self.api_key_mgr.get("GOOGLE_API_KEY"):
                    raise ValueError("GOOGLE_API_KEY is missing in environment variables")
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

            elif provider == "groq":
                if not self.api_key_mgr.get("GROQ_API_KEY"):
                    raise ValueError("GROQ_API_KEY is missing in environment variables")
                llm = ChatGroq(
                    model=model_name,
                    api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                    temperature=temperature,
                )

            elif provider == "openai":
                if not self.api_key_mgr.get("OPENAI_API_KEY"):
                    raise ValueError("OPENAI_API_KEY is missing in environment variables")
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
                    temperature=temperature,
                )

            else:
                log.error("Unsupported LLM provider encountered", provider=provider)
                raise ValueError(f"Unsupported LLM provider: {provider}")

            log.info("LLM loaded successfully", provider=provider, model=model_name)
            return llm

        except Exception as e:
            log.error("Error loading LLM", error=str(e))
            raise ResearchAnalystException("Failed to load LLM", sys)


# ----------------------------------------------------------------------
# Standalone Testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        loader = ModelLoader()

        # # Test embedding model
        # embeddings = loader.load_embeddings()
        # print(f"Embedding Model Loaded: {embeddings}")
        # result = embeddings.embed_query("Hello, how are you?")
        # print(f"Embedding Result: {result[:5]} ...")

        # Test LLM
        llm = loader.load_llm()
        print(f"LLM Loaded: {llm}")
        result = llm.invoke("Hello, how are you?")
        print(f"LLM Result: {result.content[:200]}")

        log.info("ModelLoader test completed successfully")

    except Exception as e:
        wrapped = e if isinstance(e, ResearchAnalystException) else ResearchAnalystException("ModelLoader test failed", e)
        log.error("Critical failure in ModelLoader test", error=str(wrapped))
        if "model_decommissioned" in str(e):
            print(
                "Groq returned model_decommissioned. Update `llm.groq.model_name` in "
                "`research_and_analyst/config/configuration.yaml` or set `GROQ_MODEL_NAME`/`LLM_MODEL_NAME`."
            )
        
        
# Write a clean, enterprise-grade Python module for dynamic model loading in a structured AI backend system. The system must follow clean architecture principles and separate API key management, configuration loading, and model initialization logic. Use environment variables and YAML configuration to determine which LLM provider to load. Support OpenAI, Google Gemini, and Groq chat models. Include structured logging at every stage, avoid exposing secrets, ensure async loop safety for gRPC-based embedding APIs, and wrap all failures using a custom domain exception class. Provide complete documentation, comments, error handling, and a standalone test block for local validation.
