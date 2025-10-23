# Gaussian Splatting playground in Unity

SIGGRAPH 2023 had a paper "[**3D Gaussian Splatting for Real-Time Radiance Field Rendering**](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)" by Kerbl, Kopanas, Leimkühler, Drettakis
that is really cool! Check out their website, source code repository, data sets and so on. I've decided to try to implement the realtime visualization part (i.e. the one that takes already-produced
gaussian splat "model" file) in Unity.

![Screenshot](/docs/Images/shotOverview.jpg?raw=true "Screenshot")

Everything in this repository is based on that "OG" gaussian splatting paper. Towards end of 2023, there's a ton of
[new gaussian splatting research](https://github.com/MrNeRF/awesome-3D-gaussian-splatting) coming out; _none_ of that is in this project.

:warning: Status as of 2023 December: I'm not planning any significant further developments.

:warning: The only platforms where this is known to work are the ones that use D3D12, Metal or Vulkan graphics APIs.
PC (Windows on D3D12 or Vulkan), Mac (Metal), Linux (Vulkan) should work. Anything else I have not actually tested;
it might work or it might not.
- Some virtual reality devices work (reportedly HTC Vive, Varjo Aero, Quest 3 and Quest Pro). Some others might not
  work, e.g. Apple Vision Pro. See [#17](https://github.com/aras-p/UnityGaussianSplatting/issues/17)
- Anything using OpenGL or OpenGL ES: [#26](https://github.com/aras-p/UnityGaussianSplatting/issues/26)
- WebGPU might work someday, but seems that today it does not quite have all the required graphics features yet: [#65](https://github.com/aras-p/UnityGaussianSplatting/issues/65)
- Mobile may or might not work. Some iOS devices definitely do not work ([#72](https://github.com/aras-p/UnityGaussianSplatting/issues/72)),
  some Androids do not work either ([#112](https://github.com/aras-p/UnityGaussianSplatting/issues/112))

## Usage

Download or clone this repository, open `projects/GaussianExample` as a Unity project (I use Unity 2022.3, other versions might also work),
and open `GSTestScene` scene in there.

Note that the project requires DX12 or Vulkan on Windows, i.e. **DX11 will not work**. This is **not tested at all on mobile/web**, and probably
does not work there.

<img align="right" src="docs/Images/shotAssetCreator.png" width="250px">

Next up, **create some GaussianSplat assets**: open `Tools -> Gaussian Splats -> Create GaussianSplatAsset` menu within Unity.
In the dialog, point `Input PLY/SPZ File` to your Gaussian Splat file. Currently two
file formats are supported:
- PLY format from the original 3DGS paper (in the official paper models, the correct files
  are under `point_cloud/iteration_*/point_cloud.ply`).
- [Scaniverse SPZ](https://scaniverse.com/spz) format.

Optionally there can be `cameras.json` next to it or somewhere in parent folders.

Pick desired compression options and output folder, and press "Create Asset" button. The compression even at "very low" quality setting is decently usable, e.g. 
this capture at Very Low preset is under 8MB of total size (click to see the video): \
[![Watch the video](https://img.youtube.com/vi/iccfV0YlWVI/0.jpg)](https://youtu.be/iccfV0YlWVI)

If everything was fine, there should be a GaussianSplat asset that has several data files next to it.

Since the gaussian splat models are quite large, I have not included any in this Github repo. The original
[paper github page](https://github.com/graphdeco-inria/gaussian-splatting) has a a link to
[14GB zip](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip) of their models.


In the game object that has a `GaussianSplatRenderer` script, **point the Asset field to** one of your created assets.
There are various controls on the script to debug/visualize the data, as well as a slider to move game camera into one of asset's camera
locations.

The rendering takes game object transformation matrix into account; the official gaussian splat models seem to be all rotated by about
-160 degrees around X axis, and mirrored around Z axis, so in the sample scene the object has such a transform set up.

## ExAvatar Integration (HDRP Project)

在 `projects/GaussianExample-HDRP` 中，額外增加了 ExAvatar 推論的功能：
1.  **實現了 ExAvatar 在 Unity 上的推論**：透過 ONNX Runtime 執行 ExAvatar 的 MLP 模型，實現動態高斯人物渲染。
2.  **多物件渲染範例**：場景中的物件 `GaussianSplats(1~3)` 是 `GaussianSplats` 的複製，用於展示同時渲染多個人物的情況。主要邏輯集中在 `GaussianSplats` 物件上。
3.  **多物件遮擋**：修改了 `GaussianSplatRender` 的渲染順序，允許多個 GS 物件之間進行正確的深度遮擋。
4.  **資產準備**：
    * `GaussianSplatRender` 組件需要指定一個 `.asset` 檔案 (`Assets/GaussianAssets/output_onnx.asset`)。
    * 此 `.asset` 檔案是透過這 Unity 專案內的工具 (`Tools -> Gaussian Splats -> Create GaussianSplatAsset`) 從 `Assets/StreamingAssets/output_onnx.ply` 檔案以最高品質轉換而來，代表 ExAvatar 人體模型的 **Neutral Pose** 狀態。
5.  **ExAvatar 推論腳本** (`Assets/Script/HumanGaussianInference.cs`)：
    * 此腳本負責執行 ExAvatar 的 MLP 部分。
    * 需要指定四種 ONNX MLP 模型 (`human_model_ChunkedGroupNorm_lbs.onnx`, `human_model_ChunkedGroupNorm_lbs_refine.onnx`, `human_model_ChunkedGroupNorm_no_refine.onnx`, `human_model_ChunkedGroupNorm_lbs_static.onnx`)，這些是從 ExAvatar 原始模型導出的 ONNX 格式。
    * **動畫控制**：
        * `motionFolderName` 指定了動畫數據所在的資料夾，位於 `Assets/StreamingAssets/{motionFolderName}` (目前為 `smplx_params_smoothed`)。
        * 此資料夾包含多個 `.json` 檔案，每個檔案代表一幀的 SMPL-X pose 和 expression 參數。
        * `Refine` 選項決定是否啟用 ExAvatar 的 Refine 模式。關閉可提升效能，開啟則品質較高。
        * `Frame Index` 和 `Play` 選項決定撥放進度和是否撥放(目前播放長度是寫死的)
6.  **骨架與 LBS 變形腳本** (`Assets/Script/SkeletonBuilder.cs`)：
    * 此腳本用於在執行階段生成骨架並應用 Linear Blend Skinning (LBS) 變形。
    * 執行遊戲後才會看到生成的骨架。
    * 取消勾選 `Set POSE` 後，可以手動調整骨架關節的位置和旋轉。

Additional documentation:

* [Render Pipeline Integration](/docs/render-pipeline-integration.md)
* [Editing Splats](/docs/splat-editing.md)

_That's it!_


## Write-ups

My own blog posts about all this:
* [Gaussian Splatting is pretty cool!](https://aras-p.info/blog/2023/09/05/Gaussian-Splatting-is-pretty-cool/) (2023 Sep 5)
* [Making Gaussian Splats smaller](https://aras-p.info/blog/2023/09/13/Making-Gaussian-Splats-smaller/) (2023 Sep 13)
* [Making Gaussian Splats more smaller](https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/) (2023 Sep 27)
* [Gaussian Explosion](https://aras-p.info/blog/2023/12/08/Gaussian-explosion/) (2023 Dec 8)

## Performance numbers:

"bicycle" scene from the paper, with 6.1M splats and first camera in there, rendering at 1200x797 resolution,
at "Medium" asset quality level (282MB asset file):

* Windows (NVIDIA RTX 3080 Ti):
  * Official SBIR viewer: 7.4ms (135FPS). 4.8GB VRAM usage.
  * Unity, DX12 or Vulkan: 6.8ms (147FPS) - 4.5ms rendering, 1.1ms sorting, 0.8ms splat view calc. 1.3GB VRAM usage.
* Mac (Apple M1 Max):
  * Unity, Metal: 21.5ms (46FPS).

Besides the gaussian splat asset that is loaded into GPU memory, currently this also needs about 48 bytes of GPU memory
per splat (for sorting, caching view dependent data etc.).


## License and External Code Used

The code I wrote for this is under MIT license. The project also uses several 3rd party libraries:

- [zanders3/json](https://github.com/zanders3/json), MIT license, (c) 2018 Alex Parker.
- "DeviceRadixSort" GPU sorting code contributed by Thomas Smith ([#82](https://github.com/aras-p/UnityGaussianSplatting/pull/82)).
- Virtual Reality fixes contributed by [@ninjamode](https://github.com/ninjamode) based on
  [Unity-VR-Gaussian-Splatting](https://github.com/ninjamode/Unity-VR-Gaussian-Splatting).

However, keep in mind that the [license of the original paper implementation](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)
says that the official _training_ software for the Gaussian Splats is for educational / academic / non-commercial
purpose; commercial usage requires getting license from INRIA. That is: even if this viewer / integration
into Unity is just "MIT license", you need to separately consider *how* did you get your Gaussian Splat PLY files.
