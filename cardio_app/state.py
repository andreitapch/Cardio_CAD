import streamlit as st


def key(page, name):
    return f"{page}.{name}"


def load(page, name, default=None):
    return st.session_state.get(key(page, name), default)


def store(page, name, value):
    st.session_state[key(page, name)] = value
    return value


def clear(page, name):
    st.session_state.pop(key(page, name), None)
