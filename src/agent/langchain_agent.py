#pip install langchain langchain-community langchain-ollama
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_core.prompts import PromptTemplate

# Импортируем наши инструменты из ручного агента
from agent_tools import TOOLS, TOOLS_DESCRIPTION

# Инициализируем LLM (вместо 20 строк кода с requests)
llm = ChatOllama(model="qwen2.5:3b", temperature=0.1)

# Оборачиваем функции в объекты LangChain Tool
langchain_tools = [
    Tool(
        name=name,
        func=func,
        description=f"{name}(input: str) - {desc}" # LangChain сам парсит описание
    )
    for name, func in TOOLS.items() 
    # В реальном коде описания лучше хранить отдельно, но для краткости берем из TOOLS_DESCRIPTION
    for desc in [f"Инструмент {name}"] # Заглушка, так как TOOLS_DESCRIPTION - это одна строка
]

# Упрощенный вариант создания инструментов (более правильный):
langchain_tools = [
    Tool(name="search", func=TOOLS["search"], description="Поиск в базе документов"),
    Tool(name="calc", func=TOOLS["calc"], description="Математический калькулятор"),
    Tool(name="ask_api", func=TOOLS["ask_api"], description="Готовый ответ от RAG API"),
]

# Создаем агента (ВЕСЬ ЦИКЛ ИЗ 30 СТРОК ДЕЛАЕТСЯ ОДНОЙ СТРОКОЙ!)
agent = initialize_agent(
    tools=langchain_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # Классический агент с рассуждениями
    verbose=True, # Показывает "ход рассуждений" в консоли
    max_iterations=5 # Аналог нашего max_steps
)

if __name__ == "__main__":
    q = "Найди в документах количество дней отпуска и умножь на 8."
    print(f"Запуск LangChain агента...")
    # 5. Запуск
    result = agent.run(q)
    print(f"\nИтоговый ответ: {result}")