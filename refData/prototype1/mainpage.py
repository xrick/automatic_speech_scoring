# streamlit_app.py - Streamlit 前端
import streamlit as st
import requests
import time
import json
import numpy as np
import io
from datetime import datetime
import os
from dotenv import load_dotenv

# 加載環境變量
load_dotenv()

# 設置頁面配置
st.set_page_config(
    page_title="主題性自由對話語音評測系統",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# API服務的基本URL
API_URL = "http://localhost:8000"

# 自定義CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .topic-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .result-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .metric-container {
        margin-bottom: 15px;
    }
    .metric-name {
        font-weight: bold;
    }
    .feedback-text {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .suggestion-item {
        margin-bottom: 5px;
    }
    .azure-config-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 標題
st.markdown('<div class="main-header">主題性自由對話語音評測系統</div>', unsafe_allow_html=True)

# 初始化session state
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = None
if 'image_url' not in st.session_state:
    st.session_state.image_url = None
if 'audio_id' not in st.session_state:
    st.session_state.audio_id = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'corrected_transcription' not in st.session_state:
    st.session_state.corrected_transcription = None
if 'assessment_result' not in st.session_state:
    st.session_state.assessment_result = None
if 'step' not in st.session_state:
    st.session_state.step = 1  # 1: 主題選擇, 2: 圖片查看和錄音, 3: 結果顯示
if 'azure_speech_config' not in st.session_state:
    st.session_state.azure_speech_config = {
        'key': os.getenv("AZURE_SPEECH_KEY", ""),
        'region': os.getenv("AZURE_SPEECH_REGION", "eastus"),
        'endpoint': os.getenv("AZURE_SPEECH_ENDPOINT", f"https://{os.getenv('AZURE_SPEECH_REGION', 'eastus')}.api.cognitive.microsoft.com/")
    }
if 'service_tested' not in st.session_state:
    st.session_state.service_tested = False

# Azure Speech Service 配置區
with st.sidebar:
    st.markdown("## Azure Speech Service 配置")
    
    with st.form("azure_config_form"):
        st.session_state.azure_speech_config['key'] = st.text_input(
            "Azure Speech Key",
            value=st.session_state.azure_speech_config['key'],
            type="password"
        )
        
        st.session_state.azure_speech_config['region'] = st.text_input(
            "Azure Speech Region",
            value=st.session_state.azure_speech_config['region']
        )
        
        st.session_state.azure_speech_config['endpoint'] = st.text_input(
            "Azure Speech Endpoint",
            value=st.session_state.azure_speech_config['endpoint']
        )
        
        submit_button = st.form_submit_button("儲存配置")
        
        if submit_button:
            st.success("配置已儲存!")
    
    if st.button("測試 Azure Speech 服務"):
        try:
            response = requests.get(
                f"{API_URL}/test-speech-service",
                params={
                    'speech_key': st.session_state.azure_speech_config['key'],
                    'speech_region': st.session_state.azure_speech_config['region'],
                    'speech_endpoint': st.session_state.azure_speech_config['endpoint']
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    st.success("Azure Speech 服務連接成功!")
                    st.session_state.service_tested = True
                else:
                    st.error(f"服務測試失敗: {result.get('message')}")
            else:
                st.error(f"API請求失敗: {response.status_code}")
        except Exception as e:
            st.error(f"測試過程中發生錯誤: {str(e)}")

# 主要流程
if not st.session_state.azure_speech_config['key']:
    st.warning("請先在側邊欄配置 Azure Speech 服務")
else:
    # 主題選擇區
    if st.session_state.step == 1:
        st.markdown('<div class="topic-card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">選擇對話主題</div>', unsafe_allow_html=True)
        
        # 獲取主題列表
        try:
            response = requests.get(f"{API_URL}/topics")
            if response.status_code == 200:
                topics = response.json().get("topics", [])
                
                # 下拉選單顯示主題
                selected_topic = st.selectbox("請選擇一個感興趣的主題", [""] + topics, format_func=lambda x: x if x else "請選擇...")
                
                if st.button("確定主題選擇", disabled=not selected_topic):
                    st.session_state.selected_topic = selected_topic
                    
                    # 產生相關圖片
                    with st.spinner("正在生成主題相關圖片..."):
                        response = requests.post(
                            f"{API_URL}/generate-image",
                            json={"topic": selected_topic}
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.image_url = result.get("image_url")
                            st.session_state.step = 2
                            st.experimental_rerun()
                        else:
                            st.error("生成圖片失敗，請重試")
            else:
                st.error(f"獲取主題列表失敗: {response.status_code}")
        except Exception as e:
            st.error(f"連接API服務失敗: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # 圖片顯示和錄音區
    elif st.session_state.step == 2:
        st.markdown('<div class="topic-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">主題: {st.session_state.selected_topic}</div>', unsafe_allow_html=True)
        
        # 顯示主題圖片
        if st.session_state.image_url:
            st.image(st.session_state.image_url, caption="主題相關圖片", use_column_width=True)
            st.markdown('<p>請看著圖片，自由發揮您的想法，點擊下方錄音按鈕開始錄音。</p>', unsafe_allow_html=True)
        
        # 錄音功能
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 錄音區")
            audio_data = st.audio_recorder(
                "點擊錄音", 
                pause_threshold=2.0,
                sample_rate=16000,
            )
            
            if audio_data is not None:
                st.session_state.audio_data = audio_data
                st.audio(audio_data, format="audio/wav")
        
        with col2:
            st.markdown("### 提交評測")
            if st.session_state.audio_data is not None:
                if st.button("提交評測"):
                    with st.spinner("正在處理您的錄音..."):
                        # 上傳音頻檔
                        files = {"file": ("recording.wav", st.session_state.audio_data, "audio/wav")}
                        try:
                            response = requests.post(f"{API_URL}/upload-audio", files=files)
                            
                            if response.status_code == 200:
                                audio_id = response.json().get("audio_id")
                                st.session_state.audio_id = audio_id
                                
                                # 語音轉文字
                                stt_response = requests.post(
                                    f"{API_URL}/speech-to-text",
                                    params={
                                        "audio_id": audio_id,
                                        "speech_key": st.session_state.azure_speech_config['key'],
                                        "speech_region": st.session_state.azure_speech_config['region']
                                    }
                                )
                                
                                if stt_response.status_code == 200 and stt_response.json().get("status") == "success":
                                    transcription = stt_response.json().get("text", "")
                                    st.session_state.transcription = transcription
                                    
                                    # 文字校正
                                    correction_response = requests.post(
                                        f"{API_URL}/correct-transcription",
                                        json={"text": transcription}
                                    )
                                    
                                    if correction_response.status_code == 200:
                                        corrected_text = correction_response.json().get("corrected_text", transcription)
                                        st.session_state.corrected_transcription = corrected_text
                                        
                                        # 評測語音
                                        assessment_response = requests.post(
                                            f"{API_URL}/assess-speech",
                                            json={
                                                "topic": st.session_state.selected_topic,
                                                "audio_id": audio_id,
                                                "transcription": corrected_text
                                            },
                                            params={
                                                "speech_key": st.session_state.azure_speech_config['key'],
                                                "speech_region": st.session_state.azure_speech_config['region']
                                            }
                                        )
                                        
                                        if assessment_response.status_code == 200:
                                            st.session_state.assessment_result = assessment_response.json()
                                            st.session_state.step = 3
                                            st.experimental_rerun()
                                        else:
                                            st.error(f"評測失敗: {assessment_response.text}")
                                    else:
                                        st.error(f"文字校正失敗: {correction_response.text}")
                                else:
                                    st.error(f"語音轉文字失敗: {stt_response.text}")
                            else:
                                st.error(f"上傳錄音失敗: {response.text}")
                        except Exception as e:
                            st.error(f"處理過程中發生錯誤: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # 結果顯示區
    elif st.session_state.step == 3:
        if st.session_state.assessment_result:
            result = st.session_state.assessment_result
            
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">評測結果</div>', unsafe_allow_html=True)
            
            # 顯示轉錄與校正文本
            st.markdown('<div class="sub-header" style="font-size: 1.2rem;">語音轉錄</div>', unsafe_allow_html=True)
            if st.session_state.transcription:
                st.markdown(f'<div class="feedback-text">原始轉錄: {st.session_state.transcription}</div>', unsafe_allow_html=True)
            
            if st.session_state.corrected_transcription:
                st.markdown(f'<div class="feedback-text">校正後文本: {st.session_state.corrected_transcription}</div>', unsafe_allow_html=True)
            
            # 語音質量評測
            st.markdown('<div class="sub-header" style="font-size: 1.2rem;">語音質量評測</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("發音清晰度", f"{result['speechQuality']['pronunciation']:.1f}/10")
                st.progress(result['speechQuality']['pronunciation'] / 10)
            
            with col2:
                st.metric("語速與節奏", f"{result['speechQuality']['rhythm']:.1f}/10")
                st.progress(result['speechQuality']['rhythm'] / 10)
            
            with col3:
                st.metric("語調變化", f"{result['speechQuality']['intonation']:.1f}/10")
                st.progress(result['speechQuality']['intonation'] / 10)
            
            # 內容質量評測
            st.markdown('<div class="sub-header" style="font-size: 1.2rem;">內容質量評測</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("詞彙豐富度", f"{result['contentQuality']['vocabulary']:.1f}/10")
                st.progress(result['contentQuality']['vocabulary'] / 10)
            
            with col2:
                st.metric("主題相關性", f"{result['contentQuality']['relevance']:.1f}/10")
                st.progress(result['contentQuality']['relevance'] / 10)
            
            with col3:
                st.metric("邏輯連貫性", f"{result['contentQuality']['coherence']:.1f}/10")
                st.progress(result['contentQuality']['coherence'] / 10)
            
            # 綜合評價
            st.markdown('<div class="sub-header" style="font-size: 1.2rem;">綜合評價</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="feedback-text">{result["overallFeedback"]}</div>', unsafe_allow_html=True)
            
            # 改進建議
            st.markdown('<div class="sub-header" style="font-size: 1.2rem;">改進建議</div>', unsafe_allow_html=True)
            for suggestion in result["improvementSuggestions"]:
                st.markdown(f'<div class="suggestion-item">• {suggestion}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("評測結果不可用，請重試")
        
        # 重新開始
        if st.button("重新選擇主題"):
            # 重置session state
            st.session_state.audio_data = None
            st.session_state.selected_topic = None
            st.session_state.image_url = None
            st.session_state.audio_id = None
            st.session_state.transcription = None
            st.session_state.corrected_transcription = None
            st.session_state.assessment_result = None
            st.session_state.step = 1
            st.experimental_rerun()

# 系統說明
st.sidebar.markdown("---")
st.sidebar.header("系統說明")
st.sidebar.markdown("""
### 使用流程
1. 配置 Azure Speech 服務
2. 選擇一個感興趣的主題
3. 查看系統生成的相關圖片
4. 錄制您對圖片的自由發言
5. 提交評測並查看結果

### 評測維度
- **語音質量**: 發音清晰度、語速與節奏、語調變化
- **內容質量**: 詞彙豐富度、主題相關性、邏輯連貫性

### 技術支持
- 前端: Streamlit
- 後端: FastAPI
- 語音評測: Microsoft Azure Speech Services
""")

# 顯示當前運行狀態
st.sidebar.markdown("---")
st.sidebar.markdown(f"### 系統狀態")
st.sidebar.markdown(f"當前步驟: {st.session_state.step}")
st.sidebar.markdown(f"已選主題: {st.session_state.selected_topic if st.session_state.selected_topic else '未選擇'}")
st.sidebar.markdown(f"錄音狀態: {'已完成' if st.session_state.audio_data is not None else '未錄音'}")
