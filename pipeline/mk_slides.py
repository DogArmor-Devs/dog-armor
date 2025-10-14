#!/usr/bin/env python3
import cv2
import os
import re
from pathlib import Path

def natural_sort_key(text):
    """Sort filenames naturally (handles numbers correctly)"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]

def make_video(src_folder: Path, dst_file: Path, fps: float = 1.0):
    """
    Converts all images in `src_folder` into an MP4 video at `dst_file`.
    Supported image extensions: .jpg, .jpeg, .png, .webp.
    """
    if not src_folder.is_dir():
        print(f"❌ Source folder does not exist: {src_folder}")
        return False
    
    # Gather all supported image files with natural sorting
    supported_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    imgs = [
        src_folder / f
        for f in os.listdir(src_folder)
        if (src_folder / f).is_file() and (src_folder / f).suffix.lower() in supported_extensions
    ]
    
    # Sort naturally to handle numbered sequences correctly
    imgs = sorted(imgs, key=natural_sort_key)
    
    if not imgs:
        print(f"⚠️  No supported images found in {src_folder}")
        print(f"   Looking for extensions: {', '.join(supported_extensions)}")
        return False
    
    print(f"📁 Found {len(imgs)} images in {src_folder}")
    
    # Read first image to determine video dimensions
    first_frame = cv2.imread(str(imgs[0]))
    if first_frame is None:
        print(f"❌ Cannot read first image: {imgs[0]}")
        return False
    
    height, width = first_frame.shape[:2]
    print(f"📐 Video dimensions: {width}x{height}")
    
    # Create output directory if it doesn't exist
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Try different codecs in order of preference
    codecs_to_try = ['mp4v', 'XVID', 'H264', 'MJPG']
    video_writer = None
    
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(str(dst_file), fourcc, fps, (width, height))
            if video_writer.isOpened():
                print(f"✅ Using codec: {codec}")
                break
            else:
                video_writer.release()
        except Exception as e:
            print(f"⚠️  Codec {codec} failed: {e}")
    
    if video_writer is None or not video_writer.isOpened():
        print(f"❌ Failed to initialize video writer with any codec")
        return False
    
    # Write each image as a frame
    successful_frames = 0
    for i, img_path in enumerate(imgs):
        frame = cv2.imread(str(img_path))
        if frame is not None:
            # Resize frame if dimensions don't match (optional safety check)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)
            successful_frames += 1
        else:
            print(f"⚠️  Could not read image: {img_path}")
        
        # Progress indicator for large batches
        if (i + 1) % 10 == 0 or i == len(imgs) - 1:
            print(f"   Progress: {i + 1}/{len(imgs)} images processed")
    
    video_writer.release()
    
    if successful_frames > 0:
        duration = successful_frames / fps
        print(f"✅ Saved {dst_file}")
        print(f"   📊 {successful_frames} frames at {fps} fps ({duration:.1f}s duration)")
        return True
    else:
        print(f"❌ No frames were successfully written to video")
        return False

def main():
    # Path to the parent directory containing 'front', 'side', 'top' subfolders
    base = Path("C:/Users/antoj/Documents/CS_related/Personal Projects/DogArmor_startup/dog-armor/data/processed_data/sampled")
    
    print(f"🎬 Starting image-to-video conversion")
    print(f"📂 Base directory: {base}")
    
    if not base.exists():
        print(f"❌ Base directory does not exist: {base}")
        return
    
    # Process each point-of-view folder
    views = ["front", "side", "top"]
    successful_conversions = 0
    
    for view in views:
        src_folder = base / view
        dst_video = base / f"{view}.mp4"
        
        print(f"\n{'='*50}")
        print(f"🎯 Processing '{view}' view")
        print(f"   📁 Source: {src_folder}")
        print(f"   🎥 Output: {dst_video}")
        print(f"{'='*50}")
        
        if make_video(src_folder, dst_video, fps=1.0):
            successful_conversions += 1
    
    print(f"\n🏁 Conversion complete!")
    print(f"   ✅ Successfully converted {successful_conversions}/{len(views)} folders")

if __name__ == "__main__":
    main()