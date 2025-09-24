using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using GaussianSplatting.Runtime;

public class HumanGaussianInference : MonoBehaviour
{
    public string modelPath = "human_model_smplx_beforeknn.onnx";

    [Header("Animation Data")]
    public string motionFolderName = "smplx_params_smoothed"; // 包含 smplx_params_smoothed 的資料夾
    [Range(0, 2065)]
    public int frameIndex = 0; // 您想要載入的偵數

    public GaussianSplatRenderer gaussianSplatRenderer; // 對 GaussianSplatRenderer 的引用

    private InferenceSession session;
    private readonly Dictionary<string, OrtValue> inputs = new Dictionary<string, OrtValue>();

    // 用於儲存 m_GpuOtherData 的初始狀態（包含旋轉和縮放）
    private byte[] m_InitialOtherData;
    // 快取 GPU 緩衝區的引用
    private GraphicsBuffer m_GpuOtherDataRef;

    // 與 Python 腳本中順序完全一致的 Key 列表
    private readonly List<string> smplxKeys = new List<string> {
         "body_pose", "jaw_pose", "leye_pose", "reye_pose",
        "lhand_pose", "rhand_pose", "expr"
    };

    void OnEnable()
    {
        var options = new SessionOptions();
        options.AppendExecutionProvider_DML(0); // 如果需要，啟用 GPU

        string fullModelPath = Path.Combine(Application.streamingAssetsPath, modelPath);

        try
        {
            session = new InferenceSession(fullModelPath, options);
            Debug.Log("ONNX Model loaded successfully.");
        }
        catch (OnnxRuntimeException ex)
        {
            Debug.LogError($"Failed to load ONNX model: {ex.Message}");
            return;
        }

        // 獲取初始數據
        InitializeGpuData();
    }

    private void InitializeGpuData()
    {

        if (gaussianSplatRenderer == null)
        {
            Debug.LogError("GaussianSplatRenderer 未在 Inspector 中指定！");
            return;
        }

        m_GpuOtherDataRef = gaussianSplatRenderer.GetGpuOtherData();
        if (m_GpuOtherDataRef != null)
        {
            // 讀取一次 GPU 數據並儲存起來
            m_InitialOtherData = new byte[m_GpuOtherDataRef.count * m_GpuOtherDataRef.stride];
            m_GpuOtherDataRef.GetData(m_InitialOtherData);
            Debug.Log($"成功讀取初始 'Other' 數據，大小: {m_InitialOtherData.Length} bytes。");
        }
        else
        {
            Debug.LogError("無法從 GaussianSplatRenderer 獲取 'Other' 數據緩衝區。");
        }
    }

    void Update()
    {
        // --- 從 JSON 檔案載入真實數據 ---
        LoadInputsForFrame(frameIndex);

        // --- 執行一次推論 ---
        RunInference();
    }

    // *** 這是新的核心函式 ***
    void LoadInputsForFrame(int frame)
    {
        Debug.Log($"Loading inputs for frame {frame}...");

        // --- 讀取 smplx_param ---
        string smplxFileName = $"{frame}.json";
        string smplxFilePath = Path.Combine(Application.streamingAssetsPath, motionFolderName, smplxFileName);

        if (!File.Exists(smplxFilePath))
        {
            Debug.LogError($"SMPLX file not found at: {smplxFilePath}");
            return;
        }

        // 讀取 JSON 檔案內容
        string smplxJsonContent = File.ReadAllText(smplxFilePath);
        // 使用 Newtonsoft.Json 將其解析為一個字典，其 Value 是 JToken (可以是陣列、物件等)
        var smplxData = JsonConvert.DeserializeObject<Dictionary<string, JToken>>(smplxJsonContent);

        // 遍歷 smplxKeys 列表，確保輸入順序正確
        foreach (string key in smplxKeys)
        {
            if (smplxData.ContainsKey(key))
            {
                // 將 JToken (我們知道它是陣列) 轉換為 float[]
                float[] values = smplxData[key].ToObject<float[]>();
                // Python code: torch.FloatTensor(v).view(-1)
                // C# equivalent: a flat float array, shape is just { length }
                int length = values.Length ;
                inputs[key] = CreateTensor(values, length);
            }
            else
            {
                Debug.LogWarning($"Key '{key}' not found in {smplxFileName}");
            }
        }


        Debug.Log("Inputs loaded from JSON files.");
    }

    // 一個輔助函式，方便建立 float 張量
    private OrtValue CreateTensor(float[] data, int length)
    {
        var tensor = new DenseTensor<float>(length);
        for (int i = 0; i < length; i++)
        {
            tensor[i] = (float)data[i];
        }
        long[] shape = { length };
        // 這行程式碼創建了 OrtValue，它管理著 ONNX Runtime 所需的原生記憶體
        return OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer, shape);
    }


    void RunInference()
    {
        if (session == null || inputs.Count == 0) return;
        var inputNames = inputs.Keys.ToList();
        var outputNames = session.OutputNames.ToList();
        try
        {
            using (var outputs = session.Run(new RunOptions(), inputs, outputNames))
            {
                ProcessOutputs(outputs);
            }
        }
        catch (OnnxRuntimeException ex)
        {
            Debug.LogError($"Inference failed: {ex.Message}");
        }
    }

    void ProcessOutputs(IDisposableReadOnlyCollection<OrtValue> outputs)
    {
        if (gaussianSplatRenderer == null) return;

        for (int i = 0; i < outputs.Count; i++)
        {
            var ortValue = outputs[i];
            var outputName = session.OutputNames[i];

            if (outputName == "mean_3d_refined")
            {
                var dataSpan = ortValue.GetTensorDataAsSpan<float>();
                gaussianSplatRenderer.UpdateSplatPositions(dataSpan.ToArray());
            }
            else if (outputName == "rgb")
            {
                var colorSpan = ortValue.GetTensorDataAsSpan<float>();
                float[] textureFloatData = ConvertColorsToFloatTexture(colorSpan.ToArray());
                if (textureFloatData != null)
                {
                    gaussianSplatRenderer.UpdateSplatColors(textureFloatData);
                }
            }
            else if (outputName == "scale_refined") // <-- 處理 scale 輸出
            {
                var scaleSpan = ortValue.GetTensorDataAsSpan<float>();
                byte[] updatedOtherData = PrepareOtherData(scaleSpan.ToArray());
                if (updatedOtherData != null)
                {
                    gaussianSplatRenderer.UpdateGpuOtherData(updatedOtherData);
                }
            }
        }
    }

    /// <summary>
    /// 將 ONNX輸出的 scale 數據與初始的 rotation 數據合併，
    /// 並轉換為符合 GpuOtherData 緩衝區格式的 byte[]。
    /// </summary>
    private byte[] PrepareOtherData(float[] scales)
    {
        if (m_InitialOtherData == null)
        {
            Debug.LogWarning("初始 'Other' 數據尚未準備好，跳過更新。");
            return null;
        }

        // 創建一個新的 byte 陣列，內容是初始數據的副本
        byte[] updatedData = new byte[m_InitialOtherData.Length];
        Buffer.BlockCopy(m_InitialOtherData, 0, updatedData, 0, m_InitialOtherData.Length);

        int splatCount = scales.Length / 3;
        // 對於 VeryHigh 品質，每個 splat 的 "Other" 數據由 4 bytes 旋轉 + 12 bytes 縮放組成
        int stride = m_InitialOtherData.Length / splatCount;

        for (int i = 0; i < splatCount; i++)
        {
            // 計算當前 splat 的 scale 數據在 byte 陣列中的起始位置
            // (跳過前面的 i * stride bytes，再加上 4 bytes 的旋轉數據)
            int scaleByteOffset = (i * stride) + 4;

            // 檢查邊界，防止寫入超出範圍
            if (scaleByteOffset + 12 > updatedData.Length)
            {
                Debug.LogError($"索引 {i} 超出緩衝區邊界，停止更新。");
                break;
            }

            // GPU 緩衝區中儲存的是對數縮放 (log scale)，所以我們需要將模型的線性縮放值轉換
            // 為模型輸出的 [0,1] 範圍加上一個極小值防止 log(0)
            float sx = scales[i * 3 + 0] ;
            float sy = scales[i * 3 + 1] ;
            float sz = scales[i * 3 + 2] ;

            // 將三個 float 轉換為 bytes 並寫入到 updatedData 陣列的正確位置
            Buffer.BlockCopy(BitConverter.GetBytes(sx), 0, updatedData, scaleByteOffset + 0, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(sy), 0, updatedData, scaleByteOffset + 4, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(sz), 0, updatedData, scaleByteOffset + 8, 4);
        }

        return updatedData;
    }

    // *** 新增的輔助函式 ***
    /// <summary>
    /// 將模型的 RGB 浮點數輸出轉換為 RGBA32 格式的字節陣列。
    /// </summary>
    /// <param name="floatColors">來自模型的顏色數據，假定為 [r1,g1,b1, r2,g2,b2, ...]</param>
    /// <returns>RGBA 格式的字節陣列</returns>
    // *** 修正: 新的輔助函式，用於準備完整的浮點數紋理數據 ***
    private float[] ConvertColorsToFloatTexture(float[] modelOutputColors)
    {
        if (gaussianSplatRenderer == null) return null;

        // 從 renderer 獲取正確的紋理尺寸
        int texWidth = gaussianSplatRenderer.ColorTextureWidth;
        int texHeight = gaussianSplatRenderer.ColorTextureHeight;

        if (texWidth == 0 || texHeight == 0)
        {
            Debug.LogError("Color texture dimensions from GaussianSplatRenderer are zero.");
            return null;
        }

        // 模型的輸出是 splat 數量 x 3 (RGB)
        int splatCount = modelOutputColors.Length / 3;
        // 我們需要建立一個能填滿整個紋理的陣列 (寬 x 高 x 4 個通道)
        float[] textureData = new float[texWidth * texHeight * 4];

        for (int i = 0; i < splatCount; i++)
        {
            int modelIndex = i * 3;

            /*****************************************************************
             * * 核心修改: 使用正確的索引來寫入紋理
             * 調用我們剛剛移動到 GaussianUtils.cs 的函式
             * *****************************************************************/
            int texturePixelIndex = GaussianUtils.SplatIndexToTextureIndex((uint)i);
            int textureDataIndex = texturePixelIndex * 4;

            if (textureDataIndex < textureData.Length - 4)
            {
                // 將 RGB 數據從模型輸出複製到紋理陣列的正確位置
                textureData[textureDataIndex + 0] = modelOutputColors[modelIndex + 0]; // R
                textureData[textureDataIndex + 1] = modelOutputColors[modelIndex + 1]; // G
                textureData[textureDataIndex + 2] = modelOutputColors[modelIndex + 2]; // B
                textureData[textureDataIndex + 3] = 1.0f;                               // Alpha
            }
        }

        // 陣列中剩餘的部分將預設為 0，這對於未使用的像素是安全的。

        return textureData;
    }

    void OnDestroy()
    {
        foreach (var val in inputs.Values) { val.Dispose(); }
        inputs.Clear();
        session?.Dispose();
        session = null;
    }
}