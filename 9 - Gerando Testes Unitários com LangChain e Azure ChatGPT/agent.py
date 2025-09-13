"""
agent.py
Gera automaticamente um arquivo de testes pytest (test_<module>.py)
a partir de um arquivo Python de entrada, usando LangChain + Azure OpenAI.

Uso:
    python agent.py path/to/module.py
    python agent.py path/to/module.py --run   # gera e executa pytest (opcional)
"""

import os
import argparse
import ast
import pathlib
import subprocess
from dotenv import load_dotenv

# carregar variáveis do .env (se existir)
load_dotenv()

# Import da integração AzureChatOpenAI do LangChain
# A documentação oficial mostra que a integração fica em "langchain_openai"
# e que podemos instanciar AzureChatOpenAI com azure_deployment, api_version, ...
from langchain_openai import AzureChatOpenAI


def extract_top_level_functions(py_file_path: str):
    """
    Lê um arquivo .py e retorna lista de funções top-level (nome + args + docstring).
    Usamos ast para extrair nomes de funções de forma segura (não executamos o código).
    """
    with open(py_file_path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    funcs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            arg_names = [a.arg for a in node.args.args]
            doc = ast.get_docstring(node)
            funcs.append({"name": node.name, "args": arg_names, "doc": doc})
    return funcs, src


def build_prompts(module_name: str, module_code: str, functions: list) -> tuple:
    """
    Constrói o prompt do sistema e do humano que será enviado ao LLM.
    Instruções claras ajudam o modelo a retornar apenas código Python (sem explicações).
    """
    system_prompt = (
        "Você é um assistente que produz arquivos de teste (pytest) para módulos Python. "
        "Retorne APENAS o conteúdo do arquivo de teste em Python (sem comentários, sem markdown, "
        "sem explicação). O primeiro caractere do arquivo deve começar com 'import pytest'."
    )

    # listagem simples das funções para o LLM entender o que testar
    funcs_list_text = "\n".join(
        [f"- {f['name']}({', '.join(f['args'])})" for f in functions]
    ) if functions else "(nenhuma função top-level encontrada)"

    human_prompt = f"""
Gere um arquivo pytest para o módulo "{module_name}". 
Regras obrigatórias:
1) A primeira linha do arquivo deve ser: import pytest
2) Inclua import dos simbolos testados: from {module_name} import {', '.join([f['name'] for f in functions]) if functions else '*'}
3) Para cada função top-level listada abaixo gere AO MENOS 2 testes:
   - def test_<func>_success(): teste de comportamento típico/esperado
   - def test_<func>_failure(): teste de caso limite ou exceção esperada (use pytest.raises quando apropriado)
4) Use asserts claros (assert resultado == esperado) e evite chamadas de rede, tempo ou I/O.
5) Saia apenas com código Python válido (sem explicações). 

Módulo (conteúdo):

Funções a serem testadas:
{funcs_list_text}

Gerar agora somente o conteúdo do arquivo test_{module_name}.py
"""
    return system_prompt.strip(), human_prompt.strip()


def call_azure_llm(system_prompt: str, human_prompt: str) -> str:
    """
    Chama o AzureChatOpenAI via LangChain.
    A documentação do LangChain para Azure mostra que podemos usar AzureChatOpenAI
    e invocar passando uma lista de mensagens (role, content).
    (ver docs langchain-openai / azure chat). 
    """
    # Az vars obrigatórias — garantimos que estão presentes para inicializar
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        raise RuntimeError("Defina AZURE_OPENAI_DEPLOYMENT no .env (nome do deployment).")

    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-06-01-preview")

    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        api_version=api_version,
        temperature=0.0,
        max_retries=2,
    )

    # segundo a doc, o formato de mensagens pode ser uma lista de tuplas ("system","..."), ("human","...")
    messages = [
        ("system", system_prompt),
        ("human", human_prompt),
    ]

    ai_msg = llm.invoke(messages)
    # ai_msg tem atributo .content conforme exemplo da doc
    return ai_msg.content


def ensure_imports_and_header(module_name: str, generated_code: str, functions: list) -> str:
    """
    Segurança: garantimos que o arquivo comece com 'import pytest' e contenha 'from module import ...'.
    Se o LLM falhar em incluir, inserimos nós mesmos.
    """
    lines = generated_code.splitlines()
    # garantir import pytest
    if not lines or not lines[0].strip().startswith("import pytest"):
        lines.insert(0, "import pytest")
    # garantir import das funções do módulo
    import_stmt = f"from {module_name} import {', '.join([f['name'] for f in functions])}" if functions else f"import {module_name}"
    # se a importação não existe, insere logo após a primeira linha
    if not any(line.strip().startswith(f"from {module_name} import") or line.strip().startswith(f"import {module_name}") for line in lines):
        lines.insert(1, import_stmt)
    return "\n".join(lines) + "\n"


def write_test_file(output_path: pathlib.Path, content: str):
    output_path.write_text(content, encoding="utf-8")
    print(f"[OK] Gerado: {output_path}")


def run_pytest_for_file(test_file_path: str):
    """Roda pytest para o arquivo gerado e mostra o resultado."""
    print(f"Executando pytest em {test_file_path} ...")
    completed = subprocess.run(["pytest", "-q", test_file_path], capture_output=False)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description="Gerador de testes pytest com LangChain + Azure OpenAI")
    parser.add_argument("pyfile", help="caminho para o arquivo .py que deve ser testado")
    parser.add_argument("--run", action="store_true", help="rodar pytest automaticamente após gerar")
    args = parser.parse_args()

    pyfile_path = pathlib.Path(args.pyfile)
    if not pyfile_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {pyfile_path}")

    module_name = pyfile_path.stem  # nome do arquivo sem .py

    # extrair funções e código do módulo
    functions, code = extract_top_level_functions(str(pyfile_path))

    # construir prompts
    system_prompt, human_prompt = build_prompts(module_name, code, functions)

    print("[i] Chamando LLM para gerar o teste (pode demorar alguns segundos)...")
    generated = call_azure_llm(system_prompt, human_prompt)

    # garantir 'import pytest' e 'from <module> import ...' - evita falhas do LLM
    ensured = ensure_imports_and_header(module_name, generated, functions)

    # salvar em test_<module>.py na mesma pasta do módulo
    dest = pyfile_path.parent / f"test_{module_name}.py"
    write_test_file(dest, ensured)

    if args.run:
        rc = run_pytest_for_file(str(dest))
        print(f"pytest finalizado com código de saída: {rc}")


if __name__ == "__main__":
    main()