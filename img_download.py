import pandas as pd
import pyarrow.parquet as pq
import PIL
from PIL import Image
import aiohttp
import glob
import shutil
import os
import io

table = pq.read_table('/home/c/prj/dreamgenai/the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00001-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet')
df = table.to_pandas()



async def download_file(df, i):
    dir = f"images/{df['SAMPLE_ID'][i]}"
    if os.path.exists(dir):
        return
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(df['URL'][i]) as resp:
                if resp.status == 200:
                    os.makedirs(f"{dir}")
                    # download to buffer then put into PIL
                    buffer = await resp.read()
                    image = Image.open(io.BytesIO(buffer))
                    target_width = 128
                    target_height = 128
                    
                    aspect_ratio = image.width / image.height
                    width = 128
                    height = round(width / aspect_ratio)
                    if height > 128:
                        height = 128
                        width = round(height * aspect_ratio)
                        if width > 128:
                            width = 128

                    # resize to 128X128 with black bars outside to keep same size
                    image = image.resize((width, height), PIL.Image.LANCZOS)
                    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
                    new_image.paste(image, ((target_width - width) // 2, (target_height - height) // 2))
                    # make jpg with 80% compression
                    new_image.save(f"{dir}/image.jpg", quality=80)
                    # write all metadata to info.json in same dir
                    with open(f"{dir}/info.json", "w") as f:
                        # add width and height to record
                        df.at[i, 'content_width'] = width
                        df.at[i, 'content_height'] = height
                        f.write(df.iloc[i].to_json())
        except Exception as e:
            print(e)


async def download_files():
    # find oldest dir in images, and delete in case incomplete
    if len(glob.glob('images/*')) > 0:
        oldest_dir = min(glob.glob('images/*'), key=os.path.getctime)
        shutil.rmtree(oldest_dir)

    # print first 20 rows
    for i in range(500):
        # print all cols
        print(df.iloc[i])
        await download_file(df, i)

if __name__ == "__main__":
    import asyncio
    asyncio.run(download_files())
        
