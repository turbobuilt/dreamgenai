import pandas as pd
import pyarrow.parquet as pq
import PIL
from PIL import Image
import aiohttp
import glob
import shutil
import os
import io

# table = pq.read_table('/home/c/prj/dreamgenai/the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00001-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet')
table = pq.read_table('/home/c/prj/dreamgenai/train.parquet?download=true')
df = table.to_pandas()

df.head()


import subprocess
async def download_file(df, i):
    row = df.iloc[i]
    if len(row["TEXT"]) < 10:
        print("too shor", row["TEXT"])
        return
    dir = f"images/pretty_{i}"
    if os.path.exists(dir):
        return
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(row["URL"]) as resp:
                if resp.status == 200:
                    os.makedirs(f"{dir}")
                    # download to buffer then put into PIL
                    buffer = await resp.read()

                    image = Image.open(io.BytesIO(buffer))
                    image.save(f"{dir}/image.png")
                    command = ['vipsthumbnail', 'image.png', '--smartcrop', 'attention', '-s', '128', '-o', 'image.jpg[Q=80]']
                    subprocess.run(command, check=True, cwd=dir)
                    os.remove(f"{dir}/image.png")

                    # image = Image.open(io.BytesIO(buffer))


                    # png_image_bytes = io.BytesIO()
                    # image.save(png_image_bytes, format='PNG')
                    # png_image_bytes.seek(0)
                    # command = ['vipsthumbnail', '-', '--smartcrop', 'attention', '-s', '128', '-o', 'image.jpg[Q=80]']
                    # process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    # stdout, stderr = process.communicate(input=png_image_bytes.getvalue())


                    # print("done")
                    # exit()
                    # target_width = 128
                    # target_height = 128
                    
                    # aspect_ratio = image.width / image.height
                    # width = 128
                    # height = round(width / aspect_ratio)
                    # if height > 128:
                    #     height = 128
                    #     width = round(height * aspect_ratio)
                    #     if width > 128:
                    #         width = 128

                    # # resize to 128X128 with black bars outside to keep same size
                    # image = image.resize((width, height), PIL.Image.LANCZOS)
                    # new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
                    # new_image.paste(image, ((target_width - width) // 2, (target_height - height) // 2))
                    # # make jpg with 80% compression
                    # new_image.save(f"{dir}/image.jpg", quality=80)
                    # # write all metadata to info.json in same dir
                    with open(f"{dir}/info.json", "w") as f:
                        # # add width and height to record
                        # df.at[i, 'content_width'] = width
                        # df.at[i, 'content_height'] = height
                        f.write(row.to_json())
        except Exception as e:
            print(e)


async def download_files():
    # # find oldest dir in images, and delete in case incomplete
    # if len(glob.glob('images/*')) > 0:
    #     oldest_dir = min(glob.glob('images/*'), key=os.path.getctime)
    #     shutil.rmtree(oldest_dir)

    # print first 20 rows
    for i in range(500):
        print(i)
        await download_file(df, i)

if __name__ == "__main__":
    import asyncio
    asyncio.run(download_files())
        
