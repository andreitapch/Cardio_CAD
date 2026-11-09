import streamlit as st

from cardio_app import state


def load(loader, *args, **kwargs):
    try:
        return loader(*args, **kwargs)
    except Exception as error:
        st.error(f"{loader.__name__}: {error}")
        st.stop()


def watch(page, output=None):
    job = state.load(page, "job")
    if job is None:
        return None

    @st.fragment(run_every=2 if not job.finished() else None)
    def panel():
        job.advance()
        st.caption(job.progress())
        stamp = output.stat().st_mtime if output is not None and output.is_file() else None
        if stamp is not None and stamp != state.load(page, "output_stamp"):
            state.store(page, "output_stamp", stamp)
            st.rerun()
        if job.finished() and not state.load(page, "job_seen"):
            state.store(page, "job_seen", True)
            st.rerun()

    panel()
    return job if job.finished() and not job.failed() else None


def launch(page, job):
    state.store(page, "job", job)
    state.store(page, "job_seen", False)
    state.clear(page, "result")
    state.clear(page, "output_stamp")
    st.rerun()
