<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# 角色：你是一個microsoft azure ai cognition sdk expert and MVP

問題：
請分析上傳的圖檔，並回答以下的問題：
1、圖檔中的金鑰1、金鑰2及端點的作用為何。
2、在實際用python進行撰寫連接這個服務時，圖中的「訂用帳戶識別碼」、「金鑰1」、「金鑰2」及「端點」及「位置/區域」在程式中要如何設定？
3、如何得知連接服務成功並成功計算上傳的音訊。
4、如何得知失敗？要如何進行除錯？
5、請給予一個「完整python程式碼」範例，請記住，列出"完整的程式碼"

作為Microsoft Azure AI Cognition SDK專家，我很樂意為您分析上傳的圖片並回答您的問題。

## 金鑰與端點的作用

1. **金鑰1和金鑰2**：這兩個金鑰用於驗證您的應用程式對Azure AI服務API的訪問。它們是安全憑證，允許您的應用程式與Azure Speech服務進行通信。兩個金鑰的功能完全相同，提供冗餘，使您可以在不中斷服務的情況下輪換金鑰。當您需要更新金鑰時，可以使用第二個金鑰保持服務連續性。
2. **端點**：端點是您的應用程式連接到特定Azure AI服務的URL地址。在圖中顯示為`https://eastus.api.cognitive.microsoft.com/`，這是您的應用程式發送請求的目標位置。

## Python程式中的設定方式

在Python中連接此服務時，您需要這樣設定圖中的資訊：

```python
import os
import azure.cognitiveservices.speech as speechsdk

# 設定憑證
subscription_id = "7c400507-6b30-4a2f-97f9-5baa6c9e4e28"  # 訂用帳戶識別碼
speech_key = "您的金鑰1或金鑰2"  # 從圖中的金鑰1或金鑰2填入
speech_region = "eastus"  # 位置/區域
endpoint = "https://eastus.api.cognitive.microsoft.com/"  # 端點

# 創建語音配置
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
# 如果需要使用自訂端點
# speech_config.endpoint_id = endpoint
```


## 如何確認連接成功

要確認服務連接成功並成功處理音訊：

1. **檢查回傳的結果狀態**：成功的請求會返回`ResultReason.RecognizedSpeech`狀態。
2. **檢查回傳的文字內容**：如果識別成功，會返回識別出的文字。
3. **記錄API回應**：可以記錄完整的API回應以確認處理成功。

## 如何診斷失敗

當連接或處理失敗時：

1. **檢查錯誤代碼和訊息**：Speech SDK會返回特定的錯誤代碼和訊息。
2. **檢查取消原因**：使用`result.cancellation_details`查看取消原因。
3. **記錄詳細錯誤**：記錄`error_details`和`error_code`以進行故障排除。
4. **檢查網路連接**：確保您的應用程式可以訪問Azure端點。
5. **檢查金鑰和區域**：確保使用了正確的金鑰和區域。

## 完整Python程式碼範例

以下是一個完整的Python程式碼範例，展示如何使用Azure Speech服務進行語音識別：

```python
import os
import azure.cognitiveservices.speech as speechsdk
import time

def speech_recognition_with_file():
    # 設定憑證資訊
    speech_key = "您的金鑰1或金鑰2"  # 從Azure Portal獲取
    speech_region = "eastus"  # 您的服務區域
    
    # 建立語音配置
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    
    # 設定語音識別語言（可選）
    speech_config.speech_recognition_language = "zh-TW"  # 設置為繁體中文，可根據需要更改
    
    # 指定音訊檔案路徑
    audio_file_path = "path/to/your/audio/file.wav"  # 替換為您的音訊檔案路徑
    
    # 建立音訊配置
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    
    # 建立語音識別器
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    print(f"開始處理音訊檔案: {audio_file_path}")
    
    # 開始語音識別
    result = speech_recognizer.recognize_once_async().get()
    
    # 處理結果
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"識別成功! 結果: {result.text}")
        return True, result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print(f"無法識別語音: {result.no_match_details}")
        return False, "無法識別語音"
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"語音識別已取消: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"錯誤詳情: {cancellation_details.error_details}")
            print(f"錯誤代碼: {cancellation_details.error_code}")
        return False, f"取消原因: {cancellation_details.reason}"

def continuous_recognition_with_microphone():
    # 設定憑證資訊
    speech_key = "您的金鑰1或金鑰2"  # 從Azure Portal獲取
    speech_region = "eastus"  # 您的服務區域
    
    # 建立語音配置
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = "zh-TW"
    
    # 使用麥克風作為音訊輸入
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    
    # 建立語音識別器
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    # 設定結束識別的標誌
    done = False
    
    # 定義回調函數
    def stop_cb(evt):
        print('停止連續識別')
        nonlocal done
        done = True
    
    def recognized_cb(evt):
        print(f'識別結果: {evt.result.text}')
    
    # 連接事件
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)
    
    # 開始連續識別
    speech_recognizer.start_continuous_recognition()
    print("開始說話，按Ctrl+C停止...")
    
    try:
        while not done:
            time.sleep(0.5)
    except KeyboardInterrupt:
        speech_recognizer.stop_continuous_recognition()
    
    return True, "連續識別完成"

if __name__ == "__main__":
    # 選擇要執行的功能
    print("請選擇功能:")
    print("1. 從檔案識別語音")
    print("2. 從麥克風連續識別語音")
    
    choice = input("請輸入選項 (1/2): ")
    
    if choice == "1":
        success, result = speech_recognition_with_file()
    elif choice == "2":
        success, result = continuous_recognition_with_microphone()
    else:
        print("無效的選項")
        success = False
        result = "選項錯誤"
    
    if success:
        print("程式執行成功!")
    else:
        print(f"程式執行失敗: {result}")
```

這個完整的程式碼範例提供了兩種功能：從檔案識別語音和從麥克風連續識別語音。您可以根據需要選擇使用哪一種功能，並適當修改程式碼以符合您的特定需求。

要運行此程式，請確保已安裝Azure Speech SDK：

```bash
pip install azure-cognitiveservices-speech
```

<div>⁂</div>

[^1]: https://pplx-res.cloudinary.com/image/upload/v1743846936/user_uploads/RlZZSGdPtmpiKGx/Jie-Tu-2025-04-05-Xia-Wu-5.55.17.jpg

