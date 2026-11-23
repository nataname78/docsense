"""
Client LLM unique pour DocSense.

Par defaut : le SDK Mistral. Si LLM_PROVIDER=openai : n'importe quel fournisseur
compatible avec l'API OpenAI (par exemple Gemini, Groq), choisi avec
LLM_BASE_URL, LLM_API_KEY et LLM_MODEL. L'objet renvoye a la meme interface que
le SDK Mistral (chat.complete / chat.stream), donc le reste du code ne change pas.
"""
import json
import os
from types import SimpleNamespace

import httpx
from mistralai import Mistral as _MistralSDK
from mistralai.models.sdkerror import SDKError

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class _OpenAICompatChat:
    def __init__(self, api_key, base_url):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _body(self, model, messages, temperature, stream):
        body = {"model": model, "messages": messages, "stream": stream}
        if temperature is not None:
            body["temperature"] = temperature
        effort = os.getenv("LLM_REASONING", "low")
        if effort and effort != "off":
            body["reasoning_effort"] = effort
        return body

    def complete(self, model, messages, temperature=None, max_tokens=None, **_):
        r = httpx.post(self._url, headers=self._headers, json=self._body(model, messages, temperature, False), timeout=120)
        if r.status_code >= 400:
            raise SDKError("API error occurred", r.status_code, r.text)
        content = r.json()["choices"][0]["message"].get("content") or ""
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def stream(self, model, messages, temperature=None, max_tokens=None, **_):
        with httpx.stream("POST", self._url, headers=self._headers,
                          json=self._body(model, messages, temperature, True), timeout=120) as r:
            if r.status_code >= 400:
                r.read()
                raise SDKError("API error occurred", r.status_code, r.text)
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
                except (ValueError, KeyError, IndexError):
                    continue
                yield SimpleNamespace(data=SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))]))


class _OpenAICompatClient:
    def __init__(self, api_key, base_url):
        self.chat = _OpenAICompatChat(api_key, base_url)


def Mistral(api_key=None, **kwargs):
    """Fabrique le client : SDK Mistral par defaut, ou fournisseur compatible OpenAI."""
    if os.getenv("LLM_PROVIDER", "mistral").lower() == "openai":
        key = os.getenv("LLM_API_KEY")
        if not key:
            raise RuntimeError("LLM_PROVIDER=openai mais LLM_API_KEY n'est pas defini.")
        return _OpenAICompatClient(key, os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL))
    return _MistralSDK(api_key=api_key, **kwargs)
