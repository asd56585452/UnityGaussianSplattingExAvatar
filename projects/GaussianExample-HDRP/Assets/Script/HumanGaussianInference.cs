using UnityEngine;
using Unity.InferenceEngine;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using GaussianSplatting.Runtime;
using UnityEngine.Rendering; // CommandBuffer 需要這個 namespace

public class HumanGaussianInference : MonoBehaviour
{
    // --- 靜態列表，用於追蹤所有 HumanGaussianInference 實例 ---
    public static readonly List<HumanGaussianInference> Instances = new List<HumanGaussianInference>();

    public ModelAsset modelAsset;
    Model runtimeModel;
    Worker worker;
    CommandBuffer cb;

    public ComputeShader tensorCopyShader;
    private int m_TensorCopyKernel;
    public bool cpuCopy;

    [Header("Animation Data")]
    public string motionFolderName = "smplx_params_smoothed";
    [Range(0, 2065)]
    public int frameIndex = 0;

    // --- 將 GaussianSplatRenderer 設為 public，以便從外部連結 ---
    public GaussianSplatRenderer gaussianSplatRenderer;

    // --- 新增： Tensor 屬性，用於儲存最新的位置資料 ---
    private Tensor<float> PosOutputTensor;
    private Tensor<float> RGBOutputTensor;

    private Dictionary<string, Tensor<float>> m_InputTensors = new Dictionary<string, Tensor<float>>();

    private readonly List<string> smplxKeys = new List<string> {
         "body_pose", "jaw_pose", "leye_pose", "reye_pose",
        "lhand_pose", "rhand_pose", "expr"
    };
    private readonly List<string> outputKeys = new List<string> {
        "mean_3d_refined", "rgb", "scale_refined"
    };

    void OnEnable()
    {
        // --- 將此實例加入到靜態列表中 ---
        if (!Instances.Contains(this))
        {
            Instances.Add(this);
        }

        runtimeModel = ModelLoader.Load(modelAsset);

        string firstFramePath = Path.Combine(Application.streamingAssetsPath, motionFolderName, "0.json");
        if (File.Exists(firstFramePath))
        {
            string smplxJsonContent = File.ReadAllText(firstFramePath);
            var smplxData = JsonConvert.DeserializeObject<Dictionary<string, JToken>>(smplxJsonContent);

            foreach (string key in smplxKeys)
            {
                if (smplxData.ContainsKey(key))
                {
                    float[] values = smplxData[key].ToObject<float[]>();
                    int length = values.Length;
                    // 建立 Tensor 並存入字典中
                    m_InputTensors[key] = new Tensor<float>(new TensorShape(length), values);
                }
            }
        }
        else
        {
            Debug.LogError($"無法找到第一幀的資料檔來初始化 Tensors: {firstFramePath}");
            // 你也可以在這裡根據模型的已知輸入形狀手動建立 Tensors
        }

        worker = new Worker(runtimeModel, BackendType.GPUCompute);
        LoadInputsForFrame(0,true);
        cb = new CommandBuffer();
        cb.ScheduleWorker(worker);
        Debug.Log("ONNX Model loaded successfully.");

        if (tensorCopyShader != null)
        {
            m_TensorCopyKernel = tensorCopyShader.FindKernel("CSMain");
        }
        else
        {
            Debug.LogError("TensorCopyShader 未在 InferenceRenderManager 中指定！");
        }

    }

    void OnDisable()
    {
        // --- 從靜態列表中移除此實例 ---
        Instances.Remove(this);
    }

    void Update()
    {
        // --- 從 JSON 檔案載入輸入資料 ---
        LoadInputsForFrame(frameIndex,false);

        // --- 執行推論 ---
        //worker.Schedule();
        Graphics.ExecuteCommandBuffer(cb);

        // --- 更新公開的 Tensor 屬性 ---
        PosOutputTensor = worker.PeekOutput("mean_3d_refined") as Tensor<float>;
        RGBOutputTensor = worker.PeekOutput("rgb") as Tensor<float>;
        if (cpuCopy)
        {
            var dataSpan = PosOutputTensor.DownloadToArray();
            gaussianSplatRenderer.UpdateSplatPositions(dataSpan);
            var colorSpan = RGBOutputTensor.DownloadToArray();
            float[] textureFloatData = ConvertColorsToFloatTexture(colorSpan);
            gaussianSplatRenderer.UpdateSplatColors(textureFloatData);
        }
        else
        {
            var computeTensorData = ComputeTensorData.Pin(PosOutputTensor);
            var sourceBuffer = computeTensorData.buffer;
            uint logicalElementCount = (uint)PosOutputTensor.shape.length;
            var destinationBuffer = gaussianSplatRenderer.GetGpuPosData();
            if (logicalElementCount > destinationBuffer.count)
            {
                Debug.LogError($"目標 GraphicsBuffer 的大小不足以容納 Tensor 資料！ Tensor 需要 {logicalElementCount} 個元素, 但 Buffer 只有 {destinationBuffer.count} 個。");
            }
            tensorCopyShader.SetBuffer(m_TensorCopyKernel, "_Source", sourceBuffer);
            tensorCopyShader.SetBuffer(m_TensorCopyKernel, "_Destination", destinationBuffer);
            tensorCopyShader.SetInt("_ElementCount", (int)logicalElementCount);

            int threadGroups = ((int)logicalElementCount + 63) / 64;
            tensorCopyShader.Dispatch(m_TensorCopyKernel, threadGroups, 1, 1);
        }
    }

    // LoadInputsForFrame 和其他輔助函式保持不變...
    void LoadInputsForFrame(int frame,bool SetInput)
    {
        string smplxFileName = $"{frame}.json";
        string smplxFilePath = Path.Combine(Application.streamingAssetsPath, motionFolderName, smplxFileName);

        if (!File.Exists(smplxFilePath))
        {
            // 如果檔案不存在，可以選擇跳過這一幀的更新
            return;
        }

        string smplxJsonContent = File.ReadAllText(smplxFilePath);
        var smplxData = JsonConvert.DeserializeObject<Dictionary<string, JToken>>(smplxJsonContent);

        foreach (string key in smplxKeys)
        {
            if (smplxData.ContainsKey(key) && m_InputTensors.ContainsKey(key))
            {
                float[] values = smplxData[key].ToObject<float[]>();

                // --- 核心修改：重複使用 Tensor，只更新其內部的資料 ---
                Tensor<float> tensor = m_InputTensors[key];

                // 檢查長度是否一致，以防萬一
                if (tensor.shape.length == values.Length)
                {
                    // 使用 Upload() 來更新 Tensor 的內容，這比 Download 再 Copy 更有效率
                    tensor.Upload(values);
                    if (SetInput) {
                        worker.SetInput(key, tensor);
                    }
                }
                else
                {
                    Debug.LogWarning($"Key '{key}' 的資料長度在第 {frame} 幀發生改變，無法重複使用 Tensor。");
                    // 這裡可以選擇重新建立 Tensor 作為備用方案
                }
            }
        }
    }

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
        foreach (var tensor in m_InputTensors.Values)
        {
            tensor.Dispose();
        }
        m_InputTensors.Clear();

        worker?.Dispose();
    }
}