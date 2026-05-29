import os
import urllib.request

# The elite repositories we want Craton to learn from
SOURCES = [
    # 1. Linux Kernel (The pinnacle of OS Architecture)
    "https://raw.githubusercontent.com/torvalds/linux/master/kernel/sched/core.c",
    "https://raw.githubusercontent.com/torvalds/linux/master/mm/memory.c",
    
    # 2. Redis (Mastery of High-Performance Data Structures)
    "https://raw.githubusercontent.com/redis/redis/unstable/src/dict.c",
    "https://raw.githubusercontent.com/redis/redis/unstable/src/networking.c",
    
    # 3. SQLite (Flawless, mathematically tested Database Engine logic)
    "https://raw.githubusercontent.com/sqlite/sqlite/master/src/vdbe.c",
    "https://raw.githubusercontent.com/sqlite/sqlite/master/src/btree.c"
]

def harvest():
    print("CRATON DATA HARVESTER ONLINE")
    
    # We save to the shared Downloads folder on your internal storage
    # This makes it easy for you to upload them to Google Drive later
    output_dir = os.path.expanduser("~/storage/downloads/craton_data")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Target Directory: {output_dir}")
    
    # We set a limit per chunk. For this run, we keep it small to test the pipeline.
    # In the real massive scale-up, we change this to 2 * 1024 * 1024 * 1024 (2GB)
    chunk_size_limit = 1024 * 1024 * 2 # 2 MB chunks for testing
    current_chunk = 1
    current_size = 0
    
    chunk_file_path = os.path.join(output_dir, f"knowledge_chunk_{current_chunk:02d}.c")
    chunk_file = open(chunk_file_path, "w", encoding="utf-8")
    
    for url in SOURCES:
        filename = url.split("/")[-1]
        print(f"[*] Harvesting {filename}...")
        
        try:
            # We use a user-agent to prevent GitHub from blocking our automated scraper
            req = urllib.request.Request(url, headers={'User-Agent': 'CratonHarvester/1.0'})
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8')
                
                # Write to current chunk
                chunk_file.write(content)
                chunk_file.write("\n\n/* --- CRATON SYNAPSE BREAK --- */\n\n")
                
                current_size += len(content)
                
                # If chunk exceeds limit, rotate to next chunk
                if current_size > chunk_size_limit:
                    chunk_file.close()
                    print(f"[+] Chunk {current_chunk:02d} filled ({current_size/1024/1024:.2f} MB).")
                    current_chunk += 1
                    current_size = 0
                    chunk_file_path = os.path.join(output_dir, f"knowledge_chunk_{current_chunk:02d}.c")
                    chunk_file = open(chunk_file_path, "w", encoding="utf-8")
                    
        except Exception as e:
            print(f"[!] Failed to harvest {url}: {e}")
            
    chunk_file.close()
    print("Harvest Complete. Knowledge chunks are ready in your Downloads folder.")

if __name__ == "__main__":
    harvest()
