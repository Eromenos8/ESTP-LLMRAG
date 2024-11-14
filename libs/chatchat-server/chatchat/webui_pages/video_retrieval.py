import os
import time
from typing import Dict, Literal, Tuple
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_antd_components.utils import ParseItems

from ..settings import Settings
from ..server.knowledge_base.kb_service.base import (
    get_kb_details,
    get_kb_file_details,
)
from ..server.knowledge_base.utils import LOADER_DICT, get_file_path
from ..server.utils import get_config_models, get_default_embedding
from ..webui_pages.utils import *

from VideoRetriever.video_retrieval import VideoRetriever

# SENTENCE_SIZE = 100

cell_renderer = JsCode(
    """function(params) {if(params.value==true){return '✓'}else{return '×'}}"""
)


def config_aggrid(
        df: pd.DataFrame,
        columns: Dict[Tuple[str, str], Dict] = {},
        selection_mode: Literal["single", "multiple", "disabled"] = "single",
        use_checkbox: bool = False,
) -> GridOptionsBuilder:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("No", width=40)
    for (col, header), kw in columns.items():
        gb.configure_column(col, header, wrapHeaderText=True, **kw)
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=use_checkbox,
        pre_selected_rows=st.session_state.get("selected_rows", [0]),
    )
    gb.configure_pagination(
        enabled=True, paginationAutoPageSize=False, paginationPageSize=10
    )
    return gb


def video_retrieval_page(api: ApiRequest, is_lite: bool = None):
    try:
        kb_list = {x["kb_name"]: x for x in get_kb_details()}
    except Exception as e:
        st.error(
            "Error in getting knowledge base details: {}".format(str(e))
        )
        st.stop()

    with st.sidebar:
        # slider at sidebar for top_k
        top_k = st.slider("Top k matches", 1, 10, 5)

    col1, col2 = st.columns([1, 1])

    with col1:
        # folder selector
        selected_folder = st.text_input("Choose a folder path:", key="folder_input")
        if st.button("Select Folder"):
            uploaded_files = st.file_uploader("Choose any file in the target folder")
            if uploaded_files:
                selected_folder = os.path.dirname(uploaded_files.name)
                st.session_state["folder_input"] = selected_folder

    with col2:
        # 3. Text displayer
        st.text_area("Matched Videos", value="Match videos are displayed here", height=200)

    if selected_folder:
        selected_folder = Path(str(selected_folder).strip())
        if not selected_folder.is_dir():
            st.error("Input path is not a folder!")
        else:
            retriever = VideoRetriever(video_folder_path=Path(str(selected_folder)))
            retriever.build_vec_db()

    with st.container():
        input_cols = st.columns([1, 0.2, 15, 1])

        if input_cols[0].button(":gear:"):
            st.toast("Settings button clicked!")

        if input_cols[-1].button(":wastebasket:"):
            st.toast("Clear button clicked!")

        prompt = input_cols[2].text_input("Input chat, Shift + Enter for a new line", key="prompt_input")

    if prompt and selected_folder.is_dir():
        matched_videos, similarities = retriever.find_match_videos(prompt, top_k)

        output_text = "\n".join(
            [f"Video: {video}, Similarity: {sim:.2f}" for video, sim in zip(matched_videos, similarities)]
        )
        st.text_area("Matched Videos and Similarities", value=output_text, height=300)
