Place FrailScreen test bundles here, one sub-folder per test.

Each bundle folder should contain at minimum:
  rgb_video_*.mp4
  bounding_box_data_*.json          (optional — improves quality)

Optional files:
  pose3d.json                       (cached normalized 3D pose; created/reused automatically)
  bounding_box_face_data_*.json     (TUG only)
  sppb_results.json                 (CRT — precomputed chair rise events)
  out2.pkl  /  out2.csv             (TUG/GS — precomputed phase or gait cycles)

Supported folder names include TUG, CRT, GS1/GS2, SBS, ST, and FT.
SBS/ST/FT phase tiers cover the full video/test span with in_pos or out_of_pos.

Example layout:
  input/
    NHGPAMK10001/
      10001 TUG/
        rgb_video_10001.mp4
        bounding_box_data_10001.json
    NHGPAMK20023/
      20023 CRT/
        rgb_video_20023.mp4

Run the tool (from the auto_annotate_dist folder):
  Mac / Linux:  python3 -m auto_annotate.cli
  Windows:      python  -m auto_annotate.cli

Output .eaf files will be written to the output/ folder,
mirroring the input folder structure.
