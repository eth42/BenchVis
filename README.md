# BenchVis

## Introduction
BenchVis is a small python script to display a collection of benchmark results in an interactive browser UI.  
It is primarily meant to be used to explore a multitude of measures, select the most interesting plots and export them as both pgf (latex) and csv.  
The displayed data may be collected both from single files and all *.csv files inside a directory.  
Minor configuration options should suffice to cover the most common cases like comments inside csv files etc.

## Usage
Simply put the *visualize.py* and the *config.ini* wherever your CSVs are. Checkout the options inside the *config.ini* and adapt them to match your requirements.  
Csv files are **required** to contain a header row at the very start of the file.
In case multiple csv files are read, they should have the same header row.  
To start the interactive UI, simply run *"python3 visualize.py"*.

## Subdividing measures
If you have multiple values for a certain measurement (e.g. minimum and maximum running time of some code), you can give them suffixes separated of with a prespecified character (e.g. "running_time_min", "running_time_max").
If you set the suffixes (min/max) and the separator ('_') in the *config.ini*, the interactive UI will reduce that measure ("running_time") with additional sub options.

## Configuration
The given *config.ini* contains somewhat sane default values.
As requirements around csv files vary largely, you should check out the options and adapt them to fit your project.
Here is a short overview over the available options:

**[BenchVis]**
- *base_path*: What file/directory to open at launch
- *base_path_is_rel*: Whether *base_path* is relative to the script or global
- *col_suffixes*: A list of possible measure name suffixes
- *col_suffix_sep*: The separator between measure names and suffixes
- *ignore_regex*: All lines matching this regex will be ignored when reading csv files
- *csv_extension*: The file extension to expect/use for csv files
- *csv_separator*: The separator in csv file format
- *csv_quotes*: The quotation marks in csv file format
- *csv_escape_char*: The escape character in csv file format

**[Dash]**
- *plot_height*: The height of the plot in the interactive UI in pixels
- *marker_size*: The size of the markers in the plot (see Plotly documentation for more details)
- *marker_opacity*: The opacity of the markers in the plot
- *line_width*: The line width of the markers in the plot