# czpeedy - `tensorstore` profiling tool
`czpeedy` (pronounced 'speedy') is a command-line tool used to determine the [`tensorstore`](https://github.com/google/tensorstore/) settings
(called a 'spec') which yield the fastest write speed on a given machine. For example, on some systems, it is faster
to compress data on the cpu before writing it to the comparatively slow drives. On some systems, this is not the case - to
know which is best for you, you need to perform a benchmark using the real system and the real data.

`czpeedy` can be configured with each value that you might sensibly try - compression level, codec, chunk size (important
for sequential write performance), endianness, and other properties. Then, it loads real data from your machine, and writes it
to disk using each possible combination of those parameters (which can easily be in the thousands). At the end, the average speed,
standard deviation, and fastest settings will be reported.

## Screenshots
![images/term_screenshot.png](A screenshot of the terminal output created by `czpeedy`.)