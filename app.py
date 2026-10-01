"""Checkpoint 5 - aplicação Streamlit para prever cancelamento (churn) de clientes de telecom.

Carrega o pipeline salvo pelo notebook (preparação + modelo final), sem refazer nenhuma etapa à mão.
Execução local: streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PASTA = Path(__file__).parent
LIMIAR = 0.5

ROTULOS = {
    "genero": "Gênero",
    "idoso": "Idoso (65 anos ou mais)",
    "tem_conjuge": "Tem cônjuge",
    "tem_dependentes": "Tem dependentes",
    "meses_como_cliente": "Meses como cliente",
    "telefone": "Serviço de telefone",
    "multiplas_linhas": "Múltiplas linhas",
    "internet": "Tipo de internet",
    "seguranca_online": "Segurança online",
    "backup_online": "Backup online",
    "protecao_dispositivo": "Proteção do dispositivo",
    "suporte_tecnico": "Suporte técnico",
    "streaming_tv": "Streaming de TV",
    "streaming_filmes": "Streaming de filmes",
    "contrato": "Tipo de contrato",
    "fatura_digital": "Fatura digital",
    "forma_pagamento": "Forma de pagamento",
    "valor_mensal": "Valor mensal (US$)",
    "valor_total": "Valor total já cobrado (US$)",
}
ADICIONAIS_INTERNET = ["seguranca_online", "backup_online", "protecao_dispositivo", "suporte_tecnico",
                       "streaming_tv", "streaming_filmes"]

st.set_page_config(page_title="Previsão de cancelamento", page_icon="📉", layout="wide")


@st.cache_resource
def carregar_modelo():
    return joblib.load(PASTA / "modelo_final.joblib")


modelo = carregar_modelo()
codificador = modelo.named_steps["preparacao"].named_transformers_["categoricas"]
categorias = dict(zip(codificador.feature_names_in_, codificador.categories_))
exemplos = pd.read_csv(PASTA / "exemplos_paridade.csv", index_col="indice")

st.title("Previsão de cancelamento de clientes")
st.caption(
    f"Pipeline final do notebook do Checkpoint 5 ({type(modelo.named_steps['modelo']).__name__} + one-hot). "
    "A previsão é a probabilidade de o cliente cancelar o serviço."
)

indice = st.selectbox(
    "Preencher com um cliente do teste de consistência do notebook (teste de paridade)",
    [None, *exemplos.index],
    format_func=lambda i: "Preenchimento manual" if i is None else f"Índice {i} - {exemplos.loc[i, 'caso']}",
)
exemplo = None if indice is None else exemplos.loc[indice]


def categoria(coluna, bloqueada=False):
    """Selectbox com as categorias aprendidas pelo pipeline; bloqueada quando o serviço não se aplica."""
    if bloqueada:
        return st.selectbox(ROTULOS[coluna], ["Não"], disabled=True)
    opcoes = list(categorias[coluna])
    return st.selectbox(ROTULOS[coluna], opcoes, index=0 if exemplo is None else opcoes.index(exemplo[coluna]))


def numero(coluna, padrao, **opcoes):
    return st.number_input(ROTULOS[coluna], value=padrao if exemplo is None else type(padrao)(exemplo[coluna]), **opcoes)


v = {}
perfil, servicos, conta = st.columns(3)

with perfil:
    st.subheader("Perfil")
    v["genero"] = categoria("genero")
    v["idoso"] = st.selectbox(ROTULOS["idoso"], [0, 1], index=0 if exemplo is None else int(exemplo["idoso"]),
                              format_func=lambda x: "Sim" if x else "Não")
    v["tem_conjuge"] = categoria("tem_conjuge")
    v["tem_dependentes"] = categoria("tem_dependentes")
    v["meses_como_cliente"] = numero("meses_como_cliente", 12, min_value=0, max_value=120, step=1)

with servicos:
    st.subheader("Serviços")
    v["telefone"] = categoria("telefone")
    v["multiplas_linhas"] = categoria("multiplas_linhas", bloqueada=v["telefone"] == "Não")
    v["internet"] = categoria("internet")
    for coluna in ADICIONAIS_INTERNET:
        v[coluna] = categoria(coluna, bloqueada=v["internet"] == "Não")

with conta:
    st.subheader("Conta")
    v["contrato"] = categoria("contrato")
    v["fatura_digital"] = categoria("fatura_digital")
    v["forma_pagamento"] = categoria("forma_pagamento")
    v["valor_mensal"] = numero("valor_mensal", 70.0, min_value=0.0, step=0.05, format="%.2f")
    v["valor_total"] = numero("valor_total", 840.0, min_value=0.0, step=0.05, format="%.2f")

entrada = pd.DataFrame([v])[list(modelo.feature_names_in_)]
probabilidade = float(modelo.predict_proba(entrada)[0, 1])

st.divider()
resultado, paridade = st.columns(2)

with resultado:
    st.subheader("Previsão")
    st.metric("Probabilidade de cancelamento", f"{probabilidade:.4f}")
    st.progress(probabilidade)
    if probabilidade >= LIMIAR:
        st.error(f"Previsão: o cliente **cancela** (probabilidade acima do limiar de {LIMIAR}).")
    else:
        st.success(f"Previsão: o cliente **permanece** (probabilidade abaixo do limiar de {LIMIAR}).")

with paridade:
    st.subheader("Paridade com o notebook")
    if exemplo is None:
        st.info("Selecione um cliente do teste de consistência para comparar com a previsão do notebook.")
    elif any(v[c] != exemplo[c] for c in entrada.columns):
        st.info("As entradas foram alteradas em relação ao cliente selecionado; a comparação não se aplica.")
    else:
        st.metric("Probabilidade calculada no notebook", f"{exemplo['probabilidade']:.4f}")
        st.write("Valor real:", "cancelou" if exemplo["y_real"] == 1 else "não cancelou")
        if abs(probabilidade - exemplo["probabilidade"]) < 1e-6:
            st.success("A previsão da interface coincide com a do notebook.")
        else:
            st.error("Divergência entre a interface e o notebook: investigar antes da entrega.")

with st.expander("Entradas enviadas ao pipeline"):
    st.dataframe(entrada.T.rename(columns={0: "valor"}).astype(str))
