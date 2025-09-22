using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using GaussianSplatting.Runtime;

public class HumanGaussianInference : MonoBehaviour
{
    public string modelPath = "human_model_smplx_beforeknn.onnx";

    [Header("Animation Data")]
    public string motionFolderName = "smplx_params_smoothed"; // 包含 smplx_params_smoothed 的資料夾
    public int frameIndex = 0; // 您想要載入的偵數

    private InferenceSession session;
    private readonly Dictionary<string, OrtValue> inputs = new Dictionary<string, OrtValue>();

    // 與 Python 腳本中順序完全一致的 Key 列表
    private readonly List<string> smplxKeys = new List<string> {
         "body_pose", "jaw_pose", "leye_pose", "reye_pose",
        "lhand_pose", "rhand_pose", "expr"
    };

    void Start()
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

    // 輔助函式，將 2D 陣列壓平為 1D 陣列
    private float[] Flatten2DArray(float[,] array2d)
    {
        int rows = array2d.GetLength(0);
        int cols = array2d.GetLength(1);
        float[] flat = new float[rows * cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                flat[i * cols + j] = array2d[i, j];
            }
        }
        return flat;
    }


    void RunInference()
    {
        if (session == null || inputs.Count == 0) return;
        Debug.Log("Running inference...");
        var inputNames = inputs.Keys.ToList();
        var outputNames = session.OutputNames.ToList();
        try
        {
            using (var outputs = session.Run(new RunOptions(), inputs, outputNames))
            {
                Debug.Log("Inference completed.");
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
        for (int i = 0; i < outputs.Count; i++)
        {
            var ortValue = outputs[i];
            var shape = ortValue.GetTensorTypeAndShape().Shape;
            var outputName = session.OutputNames[i];
            var dataSpan = ortValue.GetTensorDataAsSpan<float>();
            Debug.Log($"Output Name: {outputName}, Shape: [{string.Join(", ", shape)}], First 5 values: [{string.Join(", ", dataSpan.Slice(0, 5).ToArray())}]");
        }
    }

    void OnDestroy()
    {
        foreach (var val in inputs.Values) { val.Dispose(); }
        inputs.Clear();
        session?.Dispose();
        session = null;
    }
}