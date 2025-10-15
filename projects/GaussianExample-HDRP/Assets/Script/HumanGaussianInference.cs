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
    //public bool cpuCopy;

    [Header("Animation Data")]
    public string motionFolderName = "smplx_params_smoothed";
    private List<Dictionary<string, JToken>> m_AllFramesData;
    public const int FrameIndexMax = 2065;
    [Range(0, FrameIndexMax)]
    public int frameIndex = 0;
    public bool play=false;
    public float playFps=5;
    private float addPlayTime = 0;

    // --- 將 GaussianSplatRenderer 設為 public，以便從外部連結 ---
    public GaussianSplatRenderer gaussianSplatRenderer;

    // --- 新增： Tensor 屬性，用於儲存最新的位置資料 ---
    private Tensor<float> PosOutputTensor;
    public uint splatCount = 0;
    private Tensor<float> RGBOutputTensor;
    private Tensor<float> ScaleOutputTensor;
    // --- 新增： Tensor 屬性，用於儲存最新的JointPOS資料 ---
    public Tensor<float> JointZeroPoseTensor;
    public Tensor<int> ParentsTensor;
    public Tensor<float> TransformMatNeutralPoseTensor;
    public List<float> smplxPose;
    public Vector3 smplxPoseTrans;
    public bool refine = true;
    public Tensor<float> SkinningWeightTensor;

    // 用於儲存 m_GpuOtherData 的初始狀態（包含旋轉和縮放）
    private byte[] m_InitialOtherData;
    // 快取 GPU 緩衝區的引用
    private GraphicsBuffer m_GpuOtherDataRef;
    //
    public GraphicsBuffer m_GpuPosData;

    private Dictionary<string, Tensor<float>> m_InputTensors = new Dictionary<string, Tensor<float>>();

    private readonly List<string> smplxKeys = new List<string> {
         "body_pose", "jaw_pose", "leye_pose", "reye_pose",
        "lhand_pose", "rhand_pose", "expr"
    };
    private readonly List<string> outputKeys = new List<string> {
        "mean_3d_refined", "rgb", "scale_refined"
    };
    private readonly List<string> smplxPosKeys = new List<string> {
         "root_pose","body_pose", "jaw_pose", "leye_pose", "reye_pose",
        "lhand_pose", "rhand_pose"
    };
    private readonly string smplxPoseTransKey = "trans";

    void OnEnable()
    {
        // --- 將此實例加入到靜態列表中 ---
        if (!Instances.Contains(this))
        {
            Instances.Add(this);
        }

        // 預先載入所有動畫資料
        PreloadAllMotionData();

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

        InitializeGpuData();
        InitializeJointPosData();
    }

    private void InitializeJointPosData()
    {
        // --- 從 JSON 檔案載入輸入資料 ---
        LoadInputsForFrame(frameIndex, false);

        // --- 執行推論 ---
        //worker.Schedule();
        Graphics.ExecuteCommandBuffer(cb);

        // --- 更新公開的 Tensor 屬性 ---
        JointZeroPoseTensor = worker.PeekOutput("joint_zero_pose") as Tensor<float>;
        ParentsTensor = worker.PeekOutput("parents") as Tensor<int>;
        TransformMatNeutralPoseTensor = worker.PeekOutput("transform_mat_neutral_pose") as Tensor<float>;
        SkinningWeightTensor = worker.PeekOutput("skinning_weight") as Tensor<float>;
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
        //初始化未變形GpuPosData
        var sourceBuffer = gaussianSplatRenderer.GetGpuPosData();
        m_GpuPosData = new GraphicsBuffer(sourceBuffer.target, sourceBuffer.count, sourceBuffer.stride);
    }

    void OnDisable()
    {
        // --- 從靜態列表中移除此實例 ---
        Instances.Remove(this);
    }

    void Update()
    {
        // --- 從 JSON 檔案載入輸入資料 ---
        if (play)
        {
            addPlayTime += Time.deltaTime;
            frameIndex = (int) (addPlayTime * playFps);
            if(frameIndex > FrameIndexMax)
                frameIndex = FrameIndexMax;
            LoadInputsForFrame(frameIndex, false);
        }
        else
        {
            addPlayTime = (float)frameIndex / playFps;
            LoadInputsForFrame(frameIndex, false);
        }

        // --- 為每一幀重建 CommandBuffer ---
        // 建立一個臨時的 CommandBuffer 或清除舊的
        // 使用 CommandBufferPool 會更有效率，但為了簡單起見，我們先 new 一個
        using (CommandBuffer cmd = new CommandBuffer())
        {
            cmd.name = "Human Gaussian Inference";

            // 1. 將 ONNX Worker 加入 Buffer
            cmd.ScheduleWorker(worker);

            // --- 更新公開的 Tensor 屬性 ---
            if (refine)
            {
                PosOutputTensor = worker.PeekOutput("mean_3d_refined") as Tensor<float>;
                RGBOutputTensor = worker.PeekOutput("rgb") as Tensor<float>;
                ScaleOutputTensor = worker.PeekOutput("scale_refined") as Tensor<float>;
            }
            else
            {
                PosOutputTensor = worker.PeekOutput("mean_3d") as Tensor<float>;
                RGBOutputTensor = worker.PeekOutput("rgb") as Tensor<float>;
                ScaleOutputTensor = worker.PeekOutput("scale") as Tensor<float>;
            }
            

            
             
            {
                // --- GPU 複製操作 ---
                var posComputeTensorData = ComputeTensorData.Pin(PosOutputTensor);
                var rgbComputeTensorData = ComputeTensorData.Pin(RGBOutputTensor);
                var scaleComputeTensorData = ComputeTensorData.Pin(ScaleOutputTensor);

                var posSourceBuffer = posComputeTensorData.buffer;
                var rgbSourceBuffer = rgbComputeTensorData.buffer;
                var scaleSourceBuffer = scaleComputeTensorData.buffer;

                var otherDestinationBuffer = gaussianSplatRenderer.GetGpuOtherData();
                var rgbDestinationTexture = gaussianSplatRenderer.GetGpuColorData();

                splatCount = (uint)PosOutputTensor.shape.length / 3;

                if (m_GpuPosData != null && rgbDestinationTexture != null && otherDestinationBuffer != null)
                {
                    // 2. 將 Compute Shader 的參數設定加入 Buffer
                    cmd.SetComputeBufferParam(tensorCopyShader, m_TensorCopyKernel, "_SourcePos", posSourceBuffer);
                    cmd.SetComputeBufferParam(tensorCopyShader, m_TensorCopyKernel, "_SourceRGB", rgbSourceBuffer);
                    cmd.SetComputeBufferParam(tensorCopyShader, m_TensorCopyKernel, "_SourceScale", scaleSourceBuffer);
                    cmd.SetComputeBufferParam(tensorCopyShader, m_TensorCopyKernel, "_DestinationPos", m_GpuPosData);
                    cmd.SetComputeBufferParam(tensorCopyShader, m_TensorCopyKernel, "_DestinationOther", otherDestinationBuffer);
                    cmd.SetComputeTextureParam(tensorCopyShader, m_TensorCopyKernel, "_DestinationRGB", rgbDestinationTexture);
                    cmd.SetComputeIntParam(tensorCopyShader, "_SplatCount", (int)splatCount);

                    // 3. 將 Dispatch 命令加入 Buffer
                    int threadGroups = ((int)splatCount + 1023) / 1024; // 修正：常見的整數除法進位寫法
                    cmd.DispatchCompute(tensorCopyShader, m_TensorCopyKernel, threadGroups, 1, 1);
                }
                else
                {
                    Debug.LogError("一個或多個目標緩衝區為空！");
                }

                // 4. 一次性執行所有命令 (ONNX 推理 + Compute Shader 複製)
                Graphics.ExecuteCommandBuffer(cmd);
            }
        }
    }

    // 新增：預載入所有 JSON 資料到記憶體中
    private void PreloadAllMotionData()
    {
        m_AllFramesData = new List<Dictionary<string, JToken>>();

        for (int i = 0; i <= FrameIndexMax; i++) {
            string smplxFileName = $"{i}.json";
            string smplxFilePath = Path.Combine(Application.streamingAssetsPath, motionFolderName, smplxFileName);
            string smplxJsonContent = File.ReadAllText(smplxFilePath);
            Dictionary<string, JToken> smplxData = JsonConvert.DeserializeObject<Dictionary<string, JToken>>(smplxJsonContent);
            m_AllFramesData.Add(smplxData);
        }
    }

    // LoadInputsForFrame 和其他輔助函式保持不變...
    void LoadInputsForFrame(int frame,bool SetInput)
    {
        Dictionary<string, JToken> smplxData = m_AllFramesData[frame];

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

        smplxPose.Clear();

        foreach (string key in smplxPosKeys)
        {
            if (smplxData.ContainsKey(key))
            {
                float[] values = smplxData[key].ToObject<float[]>();
                smplxPose.AddRange(values);
            }
        }

        if (smplxData.ContainsKey(smplxPoseTransKey)){
            float[] values = smplxData[smplxPoseTransKey].ToObject<float[]>();
            smplxPoseTrans = new Vector3(values[0], values[1], values[2]);
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
            float sx = scales[i * 3 + 0];
            float sy = scales[i * 3 + 1];
            float sz = scales[i * 3 + 2];

            // 將三個 float 轉換為 bytes 並寫入到 updatedData 陣列的正確位置
            Buffer.BlockCopy(BitConverter.GetBytes(sx), 0, updatedData, scaleByteOffset + 0, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(sy), 0, updatedData, scaleByteOffset + 4, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(sz), 0, updatedData, scaleByteOffset + 8, 4);
        }

        return updatedData;
    }

    void OnDestroy()
    {
        foreach (var tensor in m_InputTensors.Values)
        {
            tensor.Dispose();
        }
        m_InputTensors.Clear();

        m_GpuPosData?.Release();
        m_GpuPosData = null;

        worker?.Dispose();
    }
}