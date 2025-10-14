# scripts/convert_tptp_dir.py
import os, glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from src.parsers.tptp_parser import tptp_file_to_canonical

def process_file(infile, out_dir):
    base = os.path.basename(infile)
    name = os.path.splitext(base)[0]
    outpath = os.path.join(out_dir, f"{name}.json")
    try:
        tptp_file_to_canonical(infile, outpath)
        return (infile, outpath, "ok")
    except Exception as e:
        return (infile, None, f"err:{e}")

def main(in_dir, out_dir, jobs=4):
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for ext in ("*.p", "*.tptp", "*.tstp"):
        files += glob.glob(os.path.join(in_dir, "**", ext), recursive=True)
    print(f"Found {len(files)} files to process.")
    
    completed = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(process_file, f, out_dir): f for f in files}
        for fut in as_completed(futures):
            infile, outpath, status = fut.result()
            completed += 1
            if status != "ok":
                failed += 1
                print(f"FAILED [{completed}/{len(files)}] {infile}: {status}")
            else:
                if completed % 100 == 0:  # Progress update every 100 files
                    print(f"Progress: {completed}/{len(files)} files processed ({failed} failed)")
    
    print(f"Done processing. {completed} total, {failed} failed, {completed-failed} successful.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("usage: python convert_tptp_dir.py <in_dir> <out_dir> [jobs]")
        sys.exit(1)
    in_dir = sys.argv[1]; out_dir = sys.argv[2]; jobs = int(sys.argv[3]) if len(sys.argv)>3 else 4
    main(in_dir, out_dir, jobs)