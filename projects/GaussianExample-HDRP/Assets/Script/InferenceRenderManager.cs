// InferenceRenderManager.cs

using UnityEngine;
using Unity.InferenceEngine;
using GaussianSplatting.Runtime;

public class InferenceRenderManager : MonoBehaviour
{
    // 在 Inspector 中指定你的 Compute Shader
    public ComputeShader tensorCopyShader;
    private int m_TensorCopyKernel;
    public bool cpuCopy;

    void Start()
    {
        if (tensorCopyShader != null)
        {
            m_TensorCopyKernel = tensorCopyShader.FindKernel("CSMain");
        }
        else
        {
            Debug.LogError("TensorCopyShader 未在 InferenceRenderManager 中指定！");
        }
    }

    // 使用 LateUpdate 確保在所有 Update 之後、渲染之前執行
    void LateUpdate()
    {
        if (tensorCopyShader == null) return;

        // 遍歷所有活躍的 HumanGaussianInference 實例
        foreach (var inferenceInstance in HumanGaussianInference.Instances)
        {
            var tensor = inferenceInstance.PosOutputTensor;
            var renderer = inferenceInstance.gaussianSplatRenderer;

            if (tensor == null || renderer == null)
            {
                continue;
            }

            var destinationBuffer = renderer.GetGpuPosData();
            if (destinationBuffer == null)
            {
                continue;
            }

            if (cpuCopy)
            {
                var dataSpan = tensor.DownloadToArray();
                inferenceInstance.gaussianSplatRenderer.UpdateSplatPositions(dataSpan);
            }
            else {
                var computeTensorData = ComputeTensorData.Pin(tensor);
                var sourceBuffer = computeTensorData.buffer;
                uint logicalElementCount = (uint)tensor.shape.length;

                if (logicalElementCount > destinationBuffer.count)
                {
                    Debug.LogError($"目標 GraphicsBuffer 的大小不足以容納 Tensor 資料！ Tensor 需要 {logicalElementCount} 個元素, 但 Buffer 只有 {destinationBuffer.count} 個。");
                    continue; // 繼續處理下一個實例
                }

                // 設定並派發 Compute Shader
                tensorCopyShader.SetBuffer(m_TensorCopyKernel, "_Source", sourceBuffer);
                tensorCopyShader.SetBuffer(m_TensorCopyKernel, "_Destination", destinationBuffer);
                tensorCopyShader.SetInt("_ElementCount", (int)logicalElementCount);

                int threadGroups = ((int)logicalElementCount + 63) / 64;
                tensorCopyShader.Dispatch(m_TensorCopyKernel, threadGroups, 1, 1);
            }
        }
    }
}