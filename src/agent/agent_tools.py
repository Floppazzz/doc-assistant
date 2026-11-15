import math
from datetime import datetime
from src.rag.query_rag import load_rag_assets

# Загружаем тяжелые RAG-ассеты ОДИН раз при импорте/старте
print("Загрузка векторной базы для агента...")
rag_state = load_rag_assets()

def tool_search(query: str) -> str:
    """
    Поиск релевантных фрагментов в базе документов.
    Принимает поисковый запрос (query) и возвращает найденные текстовые чанки.
    """
    # Логика поиска из query_rag.py (без вызова Ollama)
    query_vector = rag_state["embedder"].encode([query], convert_to_numpy=True)
    import faiss
    faiss.normalize_L2(query_vector)
    scores, indices = rag_state["index"].search(query_vector, k=3)
    
    chunks = []
    for idx in indices[0]:
        if idx != -1:
            chunks.append(rag_state["db_data"]["chunks"][idx])
            
    return "\n---\n".join(chunks) if chunks else "Ничего не найдено."

def tool_calc(expression: str) -> str:
    """
    Вычислить арифметическое выражение, например '2*(3+4)'.
    Принимает строку с математическим выражением и возвращает результат.
    """
    # ВНИМАНИЕ: eval используется только в учебных целях!
    # В реальном проде это критическая уязвимость (RCE).
    try:
        # Разрешаем только базовые математические операции для безопасности
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Ошибка вычисления: {e}"

def tool_get_date() -> str:
    """Возвращает текущую дату и время в формате ГГГГ-ММ-ДД."""
    return datetime.now().strftime("%Y-%m-%d")

# Реестр инструментов (Словарь: имя -> функция)
TOOLS = {
    "search": tool_search,
    "calc": tool_calc,
    "get_date": tool_get_date
}

# Описания для промпта (то, что "видит" LLM)
TOOLS_DESCRIPTION = """
1. search(query: str) - поиск информации в базе знаний.
2. calc(expression: str) - вычисление математических выражений (например, '10 * 8').
3. get_date() - получить текущую дату.
"""