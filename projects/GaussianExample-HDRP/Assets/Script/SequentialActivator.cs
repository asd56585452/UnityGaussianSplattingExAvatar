using UnityEngine;
using System.Collections;

/// <summary>
/// 依照固定的時間間隔，依序啟動（SetActive(true)）
/// 一個GameObject列表中的所有物件。
/// </summary>
public class SequentialActivator : MonoBehaviour
{
    [Header("物件設置")]
    [Tooltip("要依序啟動的物件列表")]
    public GameObject[] objectsToActivate;

    [Header("時間設置")]
    [Tooltip("每個物件啟動的間隔時間（秒）")]
    public float interval = 1.0f;

    [Header("啟動選項")]
    [Tooltip("是否在序列開始前，先將所有物件設為非啟動狀態")]
    public bool deactivateAllOnStart = true;

    [Tooltip("是否在場景開始時（Start）自動啟動序列")]
    public bool beginOnStart = true;

    // 內部狀態，防止序列重複執行
    private bool isRunning = false;

    /// <summary>
    /// Unity 的 Start() 函數
    /// </summary>
    void Start()
    {
        // 如果設置了，先在開始時停用所有物件
        if (deactivateAllOnStart)
        {
            DeactivateAllObjects();
        }

        // 如果設置了，自動開始序列
        if (beginOnStart)
        {
            StartSequence();
        }
    }

    /// <summary>
    /// 公開方法：開始啟動序列
    /// (可以由按鈕或其他腳本調用)
    /// </summary>
    public void StartSequence()
    {
        // 如果序列尚未執行，則開始
        if (!isRunning)
        {
            StartCoroutine(ActivateSequenceCoroutine());
        }
    }

    /// <summary>
    /// 核心協程：處理依序啟動和等待
    /// </summary>
    private IEnumerator ActivateSequenceCoroutine()
    {
        isRunning = true;

        // 遍歷列表中的每一個物件
        foreach (GameObject obj in objectsToActivate)
        {
            // 檢查物件是否為 null (以防萬一)
            if (obj != null)
            {
                // 啟動物件
                obj.SetActive(true);

                // 等待指定的間隔秒數
                yield return new WaitForSeconds(interval);
            }
        }

        // 序列執行完畢，重置標記
        isRunning = false;
    }

    /// <summary>
    /// 將列表中的所有物件設為非啟動狀態
    /// </summary>
    public void DeactivateAllObjects()
    {
        foreach (GameObject obj in objectsToActivate)
        {
            if (obj != null)
            {
                obj.SetActive(false);
            }
        }
    }
}