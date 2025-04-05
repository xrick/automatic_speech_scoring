<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# 接續以上的討論，在你的程式碼中，我並沒有看到你有設定「端點」，是我看錯了嗎？

您觀察得很仔細。在我提供的程式碼中確實沒有明確設定端點(endpoint)。在Azure Speech SDK中，當您提供了subscription key和region時，SDK會自動根據region構建適當的端點URL。

如果您想要明確設定端點，可以使用`endpoint`參數或`endpoint_id`屬性。以下是修改後的程式碼，明確設定了端點：

```python
import azure.cognitiveservices.speech as speechsdk

def test_speech_service(speech_key, speech_region, endpoint):
    # 建立語音配置
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    
    # 明確設定端點
    speech_config.endpoint = endpoint
    
    # 測試語音服務連接
    try:
        # 使用簡單的文字轉語音測試
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        result = speech_synthesizer.speak_text_async("測試Azure語音服務連接").get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("服務測試成功：文字轉語音完成")
            return True, "服務可用"
        else:
            print(f"服務測試失敗，原因: {result.reason}")
            return False, f"服務不可用，原因: {result.reason}"
    except Exception as e:
        print(f"服務測試發生錯誤: {str(e)}")
        return False, f"連接錯誤: {str(e)}"

# 執行測試
speech_key = "您的金鑰"
speech_region = "eastus"
endpoint = "https://eastus.api.cognitive.microsoft.com/"
success, message = test_speech_service(speech_key, speech_region, endpoint)
print(f"測試結果: {'成功' if success else '失敗'}, 訊息: {message}")
```

根據您提供的圖片，端點URL是"https://eastus.api.cognitive.microsoft.com/"，這應該與您在Azure Portal中看到的一致。

不過，需要注意的是，在大多數情況下，只提供region就足夠了，因為SDK會自動構建正確的端點URL。只有在需要使用自訂端點或特殊配置時，才需要明確設定端點。

<div>⁂</div>

[^1]: https://pplx-res.cloudinary.com/image/upload/v1743846936/user_uploads/RlZZSGdPtmpiKGx/Jie-Tu-2025-04-05-Xia-Wu-5.55.17.jpg

