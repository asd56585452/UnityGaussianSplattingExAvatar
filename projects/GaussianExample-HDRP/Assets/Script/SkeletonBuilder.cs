using UnityEngine;
using Unity.InferenceEngine;
using UnityEngine.Rendering; // CommandBuffer 需要這個 namespace

public class SkeletonBuilder : MonoBehaviour
{
    public HumanGaussianInference humanGaussianInference;

    [Header("骨架數據")]
    [Tooltip("父節點索引陣列。例如 parentArray[i] = j 代表第 i 個節點的父節點是第 j 個節點。根節點的父節點應為 -1。")]
    private int[] parentArray;

    [Tooltip("T-Pose 下各節點相對於其父節點的局部位置 (Local Position)。")]
    private Vector3[] tPoseLocalPositions;
    private Matrix4x4[] mTransformMatZeroPose;

    [Tooltip("大-Pose TransformMat。")]
    private Matrix4x4[] mTransformMatNeutralPose;

    [Tooltip("大-Pose to Pose TransformMat。")]
    private Matrix4x4[] skinningMatrix;

    public bool setPose = true;

    [Header("視覺化設定")]
    [Tooltip("是否在編輯器中繪製骨架 Gizmos")]
    public bool drawGizmos = true;
    [Tooltip("節點 Gizmos 的顏色")]
    public Color jointColor = Color.green;
    [Tooltip("節點 Gizmos 的半徑大小")]
    public float jointRadius = 0.05f;
    [Tooltip("骨骼 Gizmos 的顏色")]
    public Color boneColor = Color.white;

    // 用於儲存我們創建的所有關節 GameObject
    private GameObject[] joints;

    //LBS compute shader
    CommandBuffer cb;
    public ComputeShader LBSComputeShader;
    private int m_LBSComputeKernel;
    private ComputeBuffer m_SkinningMatrixBuffer;

    void OnEnable()
    {
        parentArray = humanGaussianInference.ParentsTensor.DownloadToArray();
        float[] fPoseLocalPositions = humanGaussianInference.JointZeroPoseTensor.DownloadToArray();
        tPoseLocalPositions = ConvertFloatToVector3Array(fPoseLocalPositions);
        float[] fTransformMatNeutralPose = humanGaussianInference.TransformMatNeutralPoseTensor.DownloadToArray();
        mTransformMatNeutralPose = ConvertFloatToMatrix4x4(fTransformMatNeutralPose);

        m_LBSComputeKernel = LBSComputeShader.FindKernel("CSMain");
    }

    void Start()
    {
        // 呼叫建立骨架的函式
        BuildSkeleton();
        TransformToNeturalPose();
    }

    void Update()
    {
        SetPose();
        SetSkinningMatrix();
        DispatchLBS();
    }

    /// <summary>
    /// 執行 LBS Compute Shader
    /// </summary>
    public void DispatchLBS()
    {
        var SkinningWeightTensorData = ComputeTensorData.Pin(humanGaussianInference.SkinningWeightTensor);
        if (LBSComputeShader == null || skinningMatrix == null || skinningMatrix.Length == 0)
        {
            Debug.LogError("LBS Compute Shader 或 Skinning Matrix 未設置！");
            return;
        }

        uint splatCount = humanGaussianInference.splatCount;
        if (splatCount == 0) return;

        // 1. 更新並設置 Skinning Matrix 數據到 GPU
        m_SkinningMatrixBuffer.SetData(skinningMatrix);

        // 2. 找到 Kernel 的索引
        int kernel = LBSComputeShader.FindKernel("CSMain");

        // 3. 設置 Shader 的輸入和輸出緩衝區
        LBSComputeShader.SetBuffer(kernel, "_SourcePos", humanGaussianInference.m_GpuPosData);
        LBSComputeShader.SetBuffer(kernel, "_SkinningWeights", SkinningWeightTensorData.buffer);
        LBSComputeShader.SetBuffer(kernel, "_SkinningMatrices", m_SkinningMatrixBuffer);

        // 從 Renderer 獲取目標位置緩衝區並設置
        LBSComputeShader.SetBuffer(kernel, "_DestinationPos", humanGaussianInference.gaussianSplatRenderer.GetGpuPosData());

        // 4. 設置其他參數
        LBSComputeShader.SetInt("_SplatCount", (int)splatCount);
        LBSComputeShader.SetInt("_JointCount", (int)joints.Length);

        // 5. 計算執行緒組數量並調度
        int threadGroups = Mathf.CeilToInt(splatCount / 1024.0f);
        LBSComputeShader.Dispatch(kernel, threadGroups, 1, 1);
    }

    private void SetSkinningMatrix()
    {
        int jointCount = joints.Length;
        skinningMatrix = new Matrix4x4[jointCount];
        for (int i = 0; i < jointCount; i++)
        {
            skinningMatrix[i] = this.transform.worldToLocalMatrix * joints[i].transform.localToWorldMatrix * mTransformMatZeroPose[i].inverse * mTransformMatNeutralPose[i];
        }
    }

    private void SetPose()
    {
        if (setPose == false) return;
        int jointCount = joints.Length;
        for (int i = 0; i < jointCount; i++)
        {
            float x = humanGaussianInference.smplxPose[i * 3 + 0];
            float y = humanGaussianInference.smplxPose[i * 3 + 1];
            float z = humanGaussianInference.smplxPose[i * 3 + 2];
            Quaternion rootRot = QuatFromRodrigues(x, y, z);
            joints[i].transform.localRotation = rootRot;
            if (parentArray[i]==-1)
            {
                joints[i].transform.localPosition = humanGaussianInference.smplxPoseTrans;
            }
        }
    }

    public void TransformToNeturalPose()
    {
        int jointCount = joints.Length;
        for (int i = 0; i < jointCount; i++)
        {
            Transform transform = joints[i].transform;
            Matrix4x4 deltaMatrix = mTransformMatNeutralPose[i].inverse;
            transform.rotation = deltaMatrix.rotation;
        }
    }

    /// <summary>
    /// 建立骨架的主要函式
    /// </summary>
    public void BuildSkeleton()
    {
        // 1. 基本的數據驗證
        if (parentArray == null || tPoseLocalPositions == null || parentArray.Length != tPoseLocalPositions.Length)
        {
            Debug.LogError("骨架數據無效！請檢查 parentArray 和 tPoseLocalPositions 的長度是否一致且不為空。");
            return;
        }

        int jointCount = parentArray.Length;
        joints = new GameObject[jointCount];
        mTransformMatZeroPose = new Matrix4x4[jointCount];
        m_SkinningMatrixBuffer = new ComputeBuffer(jointCount, sizeof(float) * 16);

        // --- 第一階段：創建所有關節物件並設定其局部位置 ---
        // 我們需要先創建所有物件，才能在下一步中設定父子關係
        for (int i = 0; i < jointCount; i++)
        {
            // 創建一個新的 GameObject 來代表關節
            joints[i] = new GameObject("Joint_" + i);
            joints[i].transform.SetParent(this.transform, false);

            // 設定此關節相對於其未來父節點的局部位置
            // 注意：此時它的父節點是場景根目錄，所以 localPosition 和 world position 暫時是相同的
            joints[i].transform.localPosition = tPoseLocalPositions[i];
        }

        // --- 第二階段：根據 parentArray 建立父子層級關係 ---
        for (int i = 0; i < jointCount; i++)
        {
            int parentIndex = parentArray[i];
            Transform parentTransform = null; // 用於傳遞給 JointGizmo

            // 如果 parentIndex 是 -1，代表這是根節點 (root)，它沒有父節點
            if (parentIndex == -1)
            {
                // 將根節點設置為 SkeletonManager 的子物件，方便管理
                joints[i].transform.SetParent(this.transform, false);
            }
            // 進行安全檢查，確保父節點索引有效
            else if (parentIndex >= 0 && parentIndex < jointCount)
            {
                // 設定父子關係
                joints[i].transform.SetParent(joints[parentIndex].transform, true);
                parentTransform = joints[parentIndex].transform;
            }
            else
            {
                Debug.LogWarning($"關節 {i} 的父節點索引 {parentIndex} 無效。");
            }
            if (drawGizmos) // 根據 drawGizmos 的設定來決定是否添加
            {
                JointGizmo gizmo = joints[i].AddComponent<JointGizmo>();
                gizmo.parentTransform = parentTransform;
                gizmo.jointColor = this.jointColor;
                gizmo.jointRadius = this.jointRadius;
                gizmo.boneColor = this.boneColor;
            }
        }
        for (int i = 0; i < jointCount; i++)
        {
            mTransformMatZeroPose[i] = this.transform.worldToLocalMatrix * joints[i].transform.localToWorldMatrix;
        }

        Debug.Log("骨架建立完成！");
    }

    public Vector3[] ConvertFloatToVector3Array(float[] floats)
    {
        if (floats.Length % 3 != 0)
        {
            Debug.LogError("The length of the float array must be a multiple of 3.");
            return null;
        }

        int vectorCount = floats.Length / 3;
        Vector3[] vectorArray = new Vector3[vectorCount];

        for (int i = 0; i < vectorCount; i++)
        {
            int floatIndex = i * 3;
            vectorArray[i] = new Vector3(
                -floats[floatIndex],// 座標系轉換: SMPL-X (右手系) -> Unity (左手系)
                floats[floatIndex + 1],
                floats[floatIndex + 2]
            );
        }

        return vectorArray;
    }

    public Matrix4x4[] ConvertFloatToMatrix4x4(float[] floats)
    {
        if (floats.Length % 16 != 0)
        {
            Debug.LogError("The length of the float array must be a multiple of 16.");
            return null;
        }

        int matrixCount = floats.Length / 16;
        Matrix4x4[] matrixArray = new Matrix4x4[matrixCount];

        for (int i = 0; i < matrixCount; i++)
        {
            int baseIndex = i * 16;
            Matrix4x4 mat = new Matrix4x4();

            mat.m00 = floats[baseIndex + 0];
            mat.m01 = floats[baseIndex + 1];
            mat.m02 = floats[baseIndex + 2];
            mat.m03 = floats[baseIndex + 3];

            // Column 1
            mat.m10 = floats[baseIndex + 4];
            mat.m11 = floats[baseIndex + 5];
            mat.m12 = floats[baseIndex + 6];
            mat.m13 = floats[baseIndex + 7];

            // Column 2
            mat.m20 = floats[baseIndex + 8];
            mat.m21 = floats[baseIndex + 9];
            mat.m22 = floats[baseIndex + 10];
            mat.m23 = floats[baseIndex + 11];

            // Column 3
            mat.m30 = floats[baseIndex + 12];
            mat.m31 = floats[baseIndex + 13];
            mat.m32 = floats[baseIndex + 14];
            mat.m33 = floats[baseIndex + 15];

            // 座標系轉換: SMPL-X (右手系) -> Unity (左手系)
            mat = ConvertRightHandedToLeftHandedMatrix(mat);

            matrixArray[i] = mat;
        }

        return matrixArray;
    }

    /// <summary>
    /// 將從 Python (右手系) 導出的 Matrix4x4 轉換為 Unity (左手系) 的 Matrix4x4
    /// </summary>
    public Matrix4x4 ConvertRightHandedToLeftHandedMatrix(Matrix4x4 rhMatrix)
    {
        // 1. 分解
        Vector3 position = rhMatrix.GetPosition();
        Quaternion rotation = rhMatrix.rotation;

        // 2. 分別轉換
        // 平移：X 軸取反
        position.x = -position.x;

        // 旋轉：X 和 W 分量取反
        rotation.x = -rotation.x;
        rotation.w = -rotation.w;

        // 3. 重新組合 (Unity 的 TRS 方法會自動處理好左手系的矩陣構建)
        Matrix4x4 lhMatrix = Matrix4x4.TRS(position, rotation, Vector3.one);

        return lhMatrix;
    }

    /// <summary>
    /// 將 SMPL-X 的軸角式向量轉換為 Unity 的 Quaternion
    /// (從官方 SMPLX.cs 腳本中借鑒)
    /// </summary>
    public static Quaternion QuatFromRodrigues(float rodX, float rodY, float rodZ)
    {
        // 座標系轉換: SMPL-X (右手系) -> Unity (左手系)
        Vector3 axis = new Vector3(-rodX, rodY, rodZ);
        float angle_rad = axis.magnitude;

        if (angle_rad < 1e-6) // 避免除以零
        {
            return Quaternion.identity;
        }

        float angle_deg = -angle_rad * Mathf.Rad2Deg;
        Vector3 axis_normalized = axis / angle_rad;

        return Quaternion.AngleAxis(angle_deg, axis_normalized);
    }

    void OnDestroy()
    {
        // 釋放緩衝區，防止記憶體洩漏
        m_SkinningMatrixBuffer?.Release();
        m_SkinningMatrixBuffer = null;
    }
}
