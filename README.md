<p align="center">
  <img src="https://github.com/royerlab/czpeedy/raw/main/images/logo.png" width="150" alt="Czpeedy logo">
</p>


# czpeedy - `tensorstore` Profiling Tool
`czpeedy` (pronounced 'speedy') is a command-line tool used to determine the [`tensorstore`](https://github.com/google/tensorstore/) settings
(called a 'spec') which yield the fastest write speed on a given machine. For example, on some systems, it is faster
to compress data on the cpu before writing it to the comparatively slow drives. On some systems, this is not the case - to
know which is best for you, you need to perform a benchmark using the real system and the real data.

`czpeedy` can be configured with each value that you might sensibly try - compression level, codec, chunk size (important
for sequential write performance), endianness, and other properties. Then, it loads real data from your machine, and writes it
to disk using each possible combination of those parameters (which can easily be in the thousands). At the end, the average speed,
standard deviation, and fastest settings will be reported.

## Screenshots
![A screenshot of the terminal output created by `czpeedy`.](images/term_screenshot.png)
(Full log ommitted for brevity)
![A screenshot of the result summary created by `czpeedy`.](images/term_screenshot_2.png)

## Installation
`czpeedy` can be installed via pip for end use, or managed with [`rye`](https://rye.astral.sh/) if you are developing for it. For most users,
we reccommend running
`pip install czpeedy`
and then following the usage instructions below. To use rye, [install it](https://rye.astral.sh/) and then use `rye run czpeedy` instead of `czpeedy`
in your shell.

## Usage
The most basic use of `czpeedy` is a write test over the entire default test space:
`czpeedy /path/to/input/file.raw --dest /path/to/output/directory --shape 1920x1080x512`
If you're willing to wait a long time (depending on the input size and drive speeds, ~a day), this command will work fine and try a
wide range of reasonable parameters. This is not an exhaustive search of all possible parameters - for example, the compression levels
tested by default max out at 5. This is because write speed usually gets bottlenecked by the cpu when high compressions are used,
so there is rarely a point in testing above 5.

If you want to specify other parameters, more cli arguments can be passed to restrict (or expand) the test
space. With the exception of `shape` and `dtype`, all parameter space adjustments can specify multiple values
by separating them with commas.

For example, if your drives are extremely slow and your cpu is extremely fast, you might want to try higher compression levels than
the defaults. To try just clevel 9, you could run:
`czpeedy /path/to/input/file.raw --dest /path/to/output/directory --shape 1920x1080x512 --clevel 9`

If you found that 9 was too high, you could specify a few more options and find the best performance as such:
`czpeedy /path/to/input/file.raw --dest /path/to/output/directory --shape 1920x1080x512 --clevel 6,7,8`

To see what other parameters you can set, invoke `czpeedy --help`. As of July 2024, it outputs the following:
```text
usage: main.py [-h] [--dest DEST] [--savecsv SAVECSV] [--repetitions REPETITIONS] [--dtype DTYPE] [--shape SHAPE] [--clevel CLEVEL] [--compressor COMPRESSOR] [--shuffle SHUFFLE] [--chunk-size CHUNK_SIZE] [--endianness ENDIANNESS] source

positional arguments:
  source                The input dataset used in benchmarking. If write benchmarking, this is the data that will be written to disk.

options:
  -h, --help            show this help message and exit
  --dest DEST           The destination where write testing will occur. A directory will be created inside, called 'czpeedy'. Each write test will delete and recreate the `czpeedy` folder.
  --savecsv SAVECSV     The destination to save test results to in csv format. Will overwrite the named file if it exists already.
  --repetitions REPETITIONS
                        The number of times to test each configuration. This increases confidence that speeds are repeatable, but takes a while. (default: 3)
  --dtype DTYPE         If your data source is a raw numpy array dump, you must provide its dtype (i.e. --dtype uint32)
  --shape SHAPE         If your data source is a raw numpy array dump, you must provide the shape (i.e. --shape 1920x1080x1024). Ignored if the data source has a shape in its metadata.
  --clevel CLEVEL       The endianness you want to write your data as (can be big, little, or none). "none" is only an acceptable endianness if the dtype is 1 byte.
  --compressor COMPRESSOR
                        The compressor id you want to use with blosc. Valid compressors: blosclz, lz4, lz4hc, snappy, zlib, zstd.
  --shuffle SHUFFLE     The shuffle mode you want to use with blosc compression. Valid shuffle types: auto, none, byte, bit
  --chunk-size CHUNK_SIZE
                        The chunk size that tensorstore should use when writing data. i.e. --chunk-size 100x100x100. Must have the same number of dimensions as the source data.
  --endianness ENDIANNESS
                        The endianness you want to write your data as (can be big, little, or none). "none" is only an acceptable endianness if the dtype is 1 byte.
  --zarr-version ZARR_VERSION
                        The version of zarr to use. (Supported: 2, 3.)
```

### Shape & Chunk Sizes
If your input source does not have included metadata about its shape (i.e. a raw numpy byte dump - as of July 2024
this is the only input type supported), you must specify the input shape using the `--shape` flag. It accepts
an `x`-delimited list of integers that match the ndarray shape. For example, if your input should have shape
`[100, 200, 300]`, then pass the argument `--shape 100x200x300`.

Because many `tensorstore` drivers use chunking, chunk shapes must be available to `czpeedy`. By default, `czpeedy` will
automatically compute a few reasonable chunk sizes with different volumes - higher chunk volumes are usually desirable
to take advantage of your disk's sequential write speed. The suggested chunk sizes will be printed out at the beginning
of program execution.

If you want to specify your own chunk sizes, you can do so using the x-delimited format described above. Additionally,
as you may want to try multiple shapes to find the best balance between compression and write speed, you can provide a
comma delimited list of chunk sizes. For example, to benchmark the chunk shape `[100, 200, 300]` against the chunk shape
`[400, 500, 600]`, you would specify `--chunk-size 100x200x300,400x500x600`. This is a bit hard to read, but is a simple
format to work with on the command line.

### Saving the Benchmark
By default, czpeedy just prints the results as they arrive, then prints details of the fastest 3 configurations
at the end of the benchmark. Because benchmarks can take a long time and may need to be interrupted, it is
highly reccommended to provide an output filepath where czpeedy can save its results as a CSV. That way,
you can easily analyze all the test data later, and `ctrl-c` without fear of losing hours of benchmark data.
To specify the output file, use `--savecsv /path/to/output.csv`.

### Test Repetitions
Disk benchmarks usually vary quite a bit between subsequent runs. To make your results more certain, `czpeedy` performs
3 repetitions of each trial it performs by default, computing the mean time and standard deviation as it goes. To use
more trials (for better data) or less trials (for faster results), use the `--repetitions` flag.
