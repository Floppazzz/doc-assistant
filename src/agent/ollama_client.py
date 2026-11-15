import requests


# OLLAMA_URL = "http://host.docker.internal:11434/v1/chat/completions"
OLLAMA_URL = "http://127.0.0.1:11434/v1/chat/completions" # Для локального запуска

def ask_ollama(prompt: str, system_prompt: str = "", temperature: float = 0.1) -> str:
    """
    Отправляет промпт в локальную модель Ollama (qwen2.5:3b) и возвращает ответ.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "qwen2.5:3b",  # Используем ту же модель, что и в Т7
        "temperature": temperature,
        "messages": messages
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"Ошибка подключения к Ollama: {e}. Убедитесь, что Ollama запущена."

if __name__ == "__main__":
    print("Проверка связи с Ollama...")
    print(ask_ollama("Назови три цвета радуги."))