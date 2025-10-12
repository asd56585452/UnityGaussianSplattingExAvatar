using UnityEngine;

public class JointGizmo : MonoBehaviour
{
    // 公開變數，讓其他腳本可以設定它們
    public Transform parentTransform;
    public Color jointColor = Color.green;
    public float jointRadius = 0.05f;
    public Color boneColor = Color.white;

    /// <summary>
    /// 在 Unity 編輯器的 Scene 視窗中繪製輔助線
    /// </summary>
    void OnDrawGizmos()
    {
        // 繪製關節點 (球體)
        // 因為這個腳本掛在關節物件上，所以 transform.position 就是關節自己的位置
        Gizmos.color = jointColor;
        Gizmos.DrawSphere(transform.position, jointRadius);

        // 如果有父節點，繪製連接到父節點的骨骼 (線段)
        if (parentTransform != null)
        {
            Gizmos.color = boneColor;
            Gizmos.DrawLine(transform.position, parentTransform.position);
        }
    }
}