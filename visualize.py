import os
import pprint
import re
import csv
import math
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import pandas as pd
import plotly.graph_objects as go
import itertools
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
import configparser

def now():
	return str(datetime.datetime.now().time())
def mpprint(o):
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(o)
def cross_filter(f, lists):
	flists = [list(x) for x in filter(lambda l: f(*l),list(zip(*lists)))]
	return [list(x) for x in zip(*flists)] if len(flists) > 0 else [[]]*len(lists)
def cross_sort(lists):
	return [list(x) for x in list(zip(*sorted(zip(*lists))))]
def str_to_bool(s):
	return s.lower() in ['yes','y','true','t','1']
def config_list(config, key, delimiter=','):
	return list(csv.reader([config[key]],delimiter=delimiter))[0]
def is_compound_key(key):
	global keys
	if not key in keys:
		return False
	return keys[key]

CONFIG_FILE = 'config.ini'
CONFIG_MAIN = 'BenchVis'
CONFIG_BASE_PATH = 'base_path'
CONFIG_BASE_PATH_IS_REL = 'base_path_is_rel'
CONFIG_SUFFIXES = 'col_suffixes'
CONFIG_SUFFIX_SEP = 'col_suffix_sep'
CONFIG_IGNORE_REGEX = 'ignore_regex'

# Read config and initialize code
config = configparser.ConfigParser()
config.read(CONFIG_FILE)
main_cfg = config[CONFIG_MAIN]
if str_to_bool(main_cfg[CONFIG_BASE_PATH_IS_REL]):
	base_path = os.path.abspath(os.path.dirname(__file__)) \
		+'/'+main_cfg[CONFIG_BASE_PATH]
else:
	base_path = main_cfg[CONFIG_BASE_PATH]
loaded_files = []
header = []
data = []
keys = {}
sub_keys = {}
suffixes = config_list(main_cfg, CONFIG_SUFFIXES)
suffix_sep = main_cfg[CONFIG_SUFFIX_SEP][1:-1]
groups = []
ignore_ident = re.compile(main_cfg[CONFIG_IGNORE_REGEX])

# Return specific (sub) key of data
def accessCol(headerKey, d):
	global keys, sub_keys, suffixes
	if headerKey in keys:
		return d[headerKey]
	else:
		for skey in sub_keys:
			if skey in headerKey:
				return d[headerKey[0:len(headerKey)-len(skey)-1]][skey]
	return None
# Write data to CSV
def writeFile(path,d):
	with open(path, 'wt', newline='') as fout:
		writer = csv.writer(fout)
		writer.writerows(d)
# Read file with header row and update keys
def readFile(path,has_header=True):
	global header,data,keys,sub_keys,suffixes,suffix_sep,ignore_ident
	print('Reading file '+path)
	first_row = has_header
	int_matcher = re.compile('^[0-9]+$')
	float_matcher = re.compile('^([0-9]+.[0-9]*)|([0-9]*.[0-9]+)$')
	with open(path, 'rt') as csvfile:
		reader = csv.reader(csvfile, skipinitialspace=True, delimiter=',', quotechar='"')
		for row in reader:
			if row == []:
				continue
			if first_row:
				header = row
				for k in header:
					for sk in suffixes:
						if k.endswith(suffix_sep+sk):
							real_key = k[0:len(k)-len(sk)-len(suffix_sep)]
							keys[real_key] = True
							if not real_key in sub_keys:
								sub_keys[real_key] = []
							if not sk in sub_keys[real_key]:
								sub_keys[real_key].append(sk)
								sub_keys[real_key] = sorted(sub_keys[real_key])
							break
					else:
						keys[k] = False
				first_row = False
			elif not ignore_ident.match(','.join(row)) is None:
				continue
			else:
				d = {}
				for k,v in zip(header,row):
					for sk in suffixes:
						if k.endswith(suffix_sep+sk):
							real_key = k[0:len(k)-len(sk)-len(suffix_sep)]
							if not real_key in d:
								d[real_key] = {}
							if int_matcher.match(v) != None:
								d[real_key][sk] = int(v)
							elif float_matcher.match(v) != None:
								d[real_key][sk] = float(v)
							else:
								d[real_key][sk] = v
							break
					else:
						if int_matcher.match(v) != None:
							d[k] = int(v)
						elif float_matcher.match(v) != None:
							d[k] = float(v)
						else:
							d[k] = v
				data.append(d)
	loaded_files.append(path)

# Read a given input path.
# If it is one file, read this file.
# If it is a directory, read all CSV file within.
def readInput(in_path):
	global loaded_files,header,data,keys,sub_keys,groups
	loaded_files = []
	header = []
	data = []
	keys = {}
	sub_keys = {}
	if os.path.isfile(in_path):
		readFile(in_path)
	elif os.path.isdir(in_path):
		for file in os.listdir(in_path):
			if file.endswith(".csv"): 
				readFile(os.path.abspath(in_path)+'/'+file)
	else:
		return
	groups = sorted([k for k in keys if not keys[k]])
	# Insert dummy column if data contains nothing
	if len(keys.keys()) == 0:
		keys['Nothing'] = False

readInput(base_path)

app = dash.Dash(__name__)
sorted_key_list = sorted(list(keys))

def make_dropdown_options(l):
	return [{'label': i, 'value': i} for i in l]
def make_dropdown(id,l):
	return dcc.Dropdown(
		id=id,
		options=make_dropdown_options(l),
		value="" if len(l) == 0 else l[0]
	)
def make_axis_type_radio(id):
	return dcc.RadioItems(
		id=id,
		options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
		value='Linear',
		labelStyle={'display': 'block', 'margin-top': '10pt'}
	)


app.layout = html.Div([
	html.Div([
		html.Div([
			html.Div([
				html.P('X-Axis'),
				make_dropdown('xaxis-column',sorted_key_list),
				make_dropdown('sub-xaxis-column',
					[] if not sorted_key_list[0] in sub_keys
					else sub_keys[sorted_key_list[0]]),
				make_axis_type_radio('xaxis-type'),
				html.P('Min val:',style={'display': 'inline-block'}),
				dcc.Input(id='x-min-input', type='text', value='', style={'min-width': '2em','display': 'inline-block', 'margin-left': '10pt', 'margin-right': '10pt'}),
				html.P('Max val:',style={'display': 'inline-block'}),
				dcc.Input(id='x-max-input', type='text', value='', style={'min-width': '2em','display': 'inline-block', 'margin-left': '10pt', 'margin-right': '10pt'})
			],
			style={'display': 'block'}),
			html.Button('Swap X and Y', id='axis-swap', n_clicks_timestamp=0, style={'margin-top': '10pt','display': 'block'}),
			html.Div([
				html.P('Y-Axis'),
				make_dropdown('yaxis-column',sorted_key_list),
				make_dropdown('sub-yaxis-column',
					[] if not sorted_key_list[0] in sub_keys
					else sub_keys[sorted_key_list[0]]),
				make_axis_type_radio('yaxis-type'),
				html.P('Min val:',style={'display': 'inline-block'}),
				dcc.Input(id='y-min-input', type='text', value='', style={'min-width': '2em','display': 'inline-block', 'margin-left': '10pt', 'margin-right': '10pt'}),
				html.P('Max val:',style={'display': 'inline-block'}),
				dcc.Input(id='y-max-input', type='text', value='', style={'min-width': '2em','display': 'inline-block', 'margin-left': '10pt', 'margin-right': '10pt'})
			],
			style={'display': 'block'})
		], style={
			'padding': '10pt',
			'display': 'inline-block',
			'vertical-align': 'top'
		}),
		html.Div([
			html.P('Group data by'),
			dcc.Checklist(
				id='group-checks',
				options=[ {'label': groups[i], 'value': i} for i in range(len(groups)) ],
				value=[],
				labelStyle={'display': 'block'}
			)
		], style={
			'padding': '10pt',
			'display': 'inline-block',
			'vertical-align': 'top'
		}),
		html.Div([
			html.P('Import data from:'),
			dcc.Input(id='in-path-input', type='text', value=base_path, style={'min-width': '35em'}),
			html.Button('Load', id='in-path-button', n_clicks_timestamp=0, style={'margin-top': '10pt', 'display': 'block'}),
			dcc.Textarea(id='loaded-files-area', readOnly=True, value='\n'.join(loaded_files), style={'min-width': '35em', 'display': 'block', 'margin-top': '10pt'}),
		], style={
			'padding': '10pt',
			'display': 'inline-block',
			'vertical-align': 'top'
		}),
		html.Div([
			html.P('Export current view to:'),
			dcc.Input(id='export-input', type='text', value=base_path, style={'min-width': '35em'}),
			html.Button('Export', id='export-button', n_clicks_timestamp=0, style={'margin-top': '10pt', 'display': 'block'}),
			html.P('Last successful export:'),
			dcc.Input(id='export-report', type='text', readOnly=True, value='Never', style={'min-width': '10em', 'display': 'inline-block'}),
		], style={
			'padding': '10pt',
			'display': 'inline-block',
			'vertical-align': 'top'
		}),
	], style={
		'borderBottom': 'thin lightgrey solid',
		'backgroundColor': 'rgb(250, 250, 250)',
		'padding': '10px 5px'
	}),
	html.Div([
		dcc.Graph(
		id='indicator-scatter'
	)], style={
		'width': '100%',
		'display': 'inline-block',
		'padding': '0 20'
	}),
	html.Div(id='axis-update',style={'display':'None'})
])

last_export_click_t = 0
last_load_click_t = 0
last_click_t = 0
last_x_val = 0
last_x_sub_val = 0
last_x_min_val = ''
last_x_max_val = ''
last_y_val = 0
last_y_sub_val = 0
last_y_min_val = ''
last_y_max_val = ''
last_x_axis_type = 0
last_y_axis_type = 0

x_data = []
y_data = []
label_data = []
data_groups = [[]]
group_names = []

# Export data button pressed
@app.callback(
	Output('export-report'		, 'value'),
	[Input('export-button'		, 'n_clicks_timestamp'),
	 Input('export-input'		, 'value')])
def export_data(click_t,path):
	global last_export_click_t, x_data, y_data, label_data, data_groups, group_names, keys
	global last_x_val, last_x_sub_val, last_y_val, last_y_sub_val, last_x_axis_type, last_y_axis_type
	if ".csv" in path or ".tex" in path:
		path = path[0:len(path)-4]
	if int(click_t) > last_export_click_t:
		last_export_click_t = int(click_t)
		dout = []
		x_key = last_x_val
		y_key = last_y_val
		if keys[last_x_val]:
			x_key = last_x_val+' '+last_x_sub_val
		if keys[last_y_val]:
			y_key = last_y_val+' '+last_y_sub_val
		dout.append(['Group',x_key,y_key])
		for i in range(len(group_names)):
			group = group_names[i]
			for x,y in zip(x_data[i],y_data[i]):
				dout.append([group,x,y])
		writeFile(path+".csv",dout)

		if last_x_axis_type == 'Linear':
			if last_y_axis_type == 'Linear':
				axis_type = 'axis'
			else:
				axis_type = 'semilogyaxis'
		else:
			if last_y_axis_type == 'Linear':
				axis_type = 'semilogxaxis'
			else:
				axis_type = 'loglogaxis'

		texData = []
		for xd,yd in zip(x_data,y_data):
			texData.append(
				'\t\t\\addplot coordinates {'+
				' '.join(['('+str(x)+','+str(y)+')' for x,y in zip(xd,yd)])+
				'};')
		
		lookupTable = {
			'Error': {
				'name': 'ERROR',
				'style': '{Red,every mark/.append style={opacity=\plotopacity,fill=Red},mark=*}'
			},
			'BlossomIV': {
				'name': 'BIV',
				'style': '{Black,every mark/.append style={opacity=\plotopacity,fill=Black},mark=*}'
			},
			'BlossomV': {
				'name': 'BV',
				'style': '{Gray,every mark/.append style={opacity=\plotopacity,fill=Gray},mark=square*}'
			},
			'EdmondsBoost': {
				'name': 'EB',
				'style': '{Peach,every mark/.append style={opacity=\plotopacity,fill=Peach},mark=*}'
			},
			'EdmondsLemon': {
				'name': 'EL',
				'style': '{Apricot,every mark/.append style={opacity=\plotopacity,fill=Apricot},mark=square*}'
			},
			'MetaGraphs': {
				'name': 'MG',
				'style': '{NavyBlue,every mark/.append style={opacity=\plotopacity,fill=NavyBlue},mark=*}'
			},
			'MetaGraphsQPT': {
				'name': 'MGL',
				'style': '{Cerulean,every mark/.append style={opacity=\plotopacity,fill=Cerulean},mark=square*}'
			},
			'MetaGraphsWR': {
				'name': 'MGB',
				'style': '{ProcessBlue,every mark/.append style={opacity=\plotopacity,fill=ProcessBlue},mark=triangle*}'
			},
			'MultiTrees': {
				'name': 'MT',
				'style': '{TealBlue,every mark/.append style={opacity=\plotopacity,fill=TealBlue},mark=*}'
			}
		}
		
		entries = []
		for group in group_names:
			for k,v in lookupTable.items():
				if k in group:
					entries.append(v)
					break
			else:
				entries.append(lookupTable['Error'])
			
		texCode = '\n'.join([
			'\\begin{tikzpicture}',
				'\t\pgfplotsset{every axis/.append style={',
					'\t\ttick style={line width=0.8pt}}}',
				'\t\\newcommand{\\plotToBeNamed}[2]{',
				'\t\\begin{'+axis_type+'}[',
					'\t\twidth=#1,',
					'\t\theight=#2,',
					'\t\tlegend style={at={(1.03,1)},anchor=north west},',
					'\t\ttitle={\\TODO{Set title}},',
					'\t\tgrid=both,',
					'\t\tgrid style={line width=.1pt, draw=gray!10},',
					'\t\tmajor grid style={line width=.2pt,draw=gray!50},',
					'\t\tcycle list={',
						'\t\t\t'+',\n\t\t\t'.join([e['style'] for e in entries]),
					'\t\t}',
					'\t\tlegend style={at={(1.03,1)},anchor=north west},',
					'\t\txlabel=\\textsc{'+x_key.replace('#','\\#')+'},',
					'\t\tylabel=\\textsc{'+y_key.replace('#','\\#')+'},',
					'\t\t]',
					'\t\t'+'\n\t\t'.join(texData),
					'\t\t\\legend{',
						'\t\t\t'+',\t\t\t'.join([e['name'] for e in entries]),
					'}',
				'\t\\end{'+axis_type+'}',
			'\\end{tikzpicture}',
			'}'
		])
		with open(path+".tex","w+") as texFile:
			texFile.write(texCode)
		return now()
	raise dash.exceptions.PreventUpdate()

@app.callback(
	[Output('xaxis-column'		, 'value'),
	 Output('sub-xaxis-column'	, 'options'),
	 Output('sub-xaxis-column'	, 'value'),
	 Output('yaxis-column'		, 'value'),
	 Output('sub-yaxis-column'	, 'options'),
	 Output('sub-yaxis-column'	, 'value'),
	 Output('x-min-input'		, 'value'),
	 Output('x-max-input'		, 'value'),
	 Output('y-min-input'		, 'value'),
	 Output('y-max-input'		, 'value'),
	 Output('xaxis-type'		, 'value'),
	 Output('yaxis-type'		, 'value')],
	[Input('axis-update'		, 'data-load'),
	 Input('axis-update'		, 'data-swap'),
	 Input('axis-update'		, 'data-col-sel')])
def update_axis_options(*dump):
	global last_x_val,last_x_sub_val,last_y_val,last_y_sub_val
	global last_x_min_val, last_x_max_val
	global last_y_min_val, last_y_max_val
	global last_x_axis_type, last_y_axis_type
	if keys[last_x_val]:
		x_opts = make_dropdown_options(sub_keys[last_x_val])
		last_x_sub_val = sub_keys[last_x_val][0]
	else:
		x_opts = []
		last_x_sub_val = None
	if keys[last_y_val]:
		y_opts = make_dropdown_options(sub_keys[last_y_val])
		last_y_sub_val = sub_keys[last_y_val][0]
	else:
		y_opts = []
		last_y_sub_val = None
	return (
		last_x_val,
		x_opts,
		last_x_sub_val,
		last_y_val,
		y_opts,
		last_y_sub_val,
		last_x_min_val, last_x_max_val,
		last_y_min_val, last_y_max_val,
		last_x_axis_type, last_y_axis_type
	)

# Load data button pressed
@app.callback(
	[Output('axis-swap'			, 'n_clicks_timestamp'),
	 Output('loaded-files-area'	, 'value'),
	 Output('xaxis-column'		, 'options'),
	 Output('yaxis-column'		, 'options'),
	 Output('axis-update'		, 'data-load'),
	 Output('group-checks'		, 'options'),
	 Output('group-checks'		, 'value')],
	[Input('in-path-button'		, 'n_clicks_timestamp'),
	 Input('in-path-input'		, 'value')])
def load_data(click_t,path):
	global loaded_files,last_load_click_t,keys,sub_keys,groups
	global last_click_t,last_x_val,last_x_sub_val,last_y_val,last_y_sub_val
	global sorted_key_list
	if int(click_t) > last_load_click_t:
		last_load_click_t = int(click_t)
		readInput(path)
		sorted_key_list = sorted(list(keys))
		first_key = sorted_key_list[0]
		last_x_val = last_y_val = first_key
		last_x_sub_val = last_y_sub_val = \
			None if not keys[first_key] \
			else sub_keys[first_key][0]
		return (last_click_t,
			'\n'.join(loaded_files),
			make_dropdown_options(sorted_key_list),
			make_dropdown_options(sorted_key_list),
			"Loaded Files",
			[{'label': groups[i], 'value': i} for i in range(len(groups))],
			[])
	raise dash.exceptions.PreventUpdate()

# Swap axis button pressed
@app.callback(
	Output('axis-update'		, 'data-swap'),
	[Input('axis-swap'			, 'n_clicks_timestamp')])
def swap_axis(click_t):
	global last_click_t
	global last_x_val, last_x_sub_val, last_y_val, last_y_sub_val
	global last_x_min_val, last_x_max_val
	global last_y_min_val, last_y_max_val
	global last_x_axis_type, last_y_axis_type
	if int(click_t) > last_click_t:
		last_click_t = int(click_t)
		last_x_val, last_y_val = last_y_val, last_x_val
		last_x_sub_val, last_y_sub_val = last_y_sub_val, last_x_sub_val
		last_x_min_val, last_y_min_val = last_y_min_val, last_x_min_val
		last_x_max_val, last_y_max_val = last_y_max_val, last_x_max_val
		last_x_axis_type, last_y_axis_type = last_y_axis_type, last_x_axis_type
		return "Swapped axes"
	raise dash.exceptions.PreventUpdate()

@app.callback(
	Output('axis-update'		, 'data-col-sel'),
	[Input('xaxis-column'		, 'value'),
	 Input('yaxis-column'		, 'value')])
def selected_column(x_val,y_val):
	global last_x_val, last_y_val
	update = False
	if x_val != last_x_val:
		last_x_val = x_val
		update = True
	if y_val != last_y_val:
		last_y_val = y_val
		update = True
	if update:
		return "Selected different column"
	raise dash.exceptions.PreventUpdate()

# Data view selection changed
@app.callback(
	Output('indicator-scatter'	, 'figure'),
	[Input('sub-xaxis-column'	, 'value'),
	 Input('x-min-input'		, 'value'),
	 Input('x-max-input'		, 'value'),
	 Input('sub-yaxis-column'	, 'value'),
	 Input('y-min-input'		, 'value'),
	 Input('y-max-input'		, 'value'),
	 Input('xaxis-type'			, 'value'),
	 Input('yaxis-type'			, 'value'),
	 Input('group-checks'		, 'value'),
	 Input('loaded-files-area'	, 'value')])
def update_graph(	xaxis_column_sub_name,
					x_min_val,
					x_max_val,
					yaxis_column_sub_name,
					y_min_val,
					y_max_val,
					xaxis_type,
					yaxis_type,
					group_checks,
					loaded_files):
	global data,keys,sub_keys,groups,suffix_sep
	global last_x_val, last_x_sub_val, last_x_min_val, last_x_max_val, last_y_val, last_y_sub_val, last_y_min_val, last_y_max_val, last_x_axis_type, last_y_axis_type
	global x_data,y_data,label_data,data_groups,group_names
	last_x_sub_val = xaxis_column_sub_name
	last_x_min_val = x_min_val
	last_x_max_val = x_max_val
	last_y_sub_val = yaxis_column_sub_name
	last_y_min_val = y_min_val
	last_y_max_val = y_max_val
	last_x_axis_type = xaxis_type
	last_y_axis_type = yaxis_type
	group_checks = sorted(group_checks)
	group_vals = {}
	for g in [groups[i] for i in group_checks]:
		group_vals[g] = sorted(list(set([str(d[g]) for d in data])))
	data_groups = [data]
	for g in [groups[i] for i in group_checks]:
		data_groups = [[d for d in dgp[0] if str(d[g]) == dgp[1]] for dgp in itertools.product(data_groups,group_vals[g])]
	data_groups = [dg for dg in data_groups if dg != []]
	x_data, y_data, label_data = [],[],[]
	if keys[last_x_val]:
		# composite key
		x_data = [[d[last_x_val][xaxis_column_sub_name] for d in data] for data in data_groups]
		x_title = last_x_val+suffix_sep+xaxis_column_sub_name
		disable_x = False
	else:
		# normal key
		x_data = [[d[last_x_val] for d in data] for data in data_groups]
		x_title = last_x_val
		disable_x = True
	if keys[last_y_val]:
		# composite key
		y_data = [[d[last_y_val][yaxis_column_sub_name] for d in data] for data in data_groups]
		y_title = last_y_val+suffix_sep+yaxis_column_sub_name
		disable_y = False
	else:
		# normal key
		y_data = [[d[last_y_val] for d in data] for data in data_groups]
		y_title = last_y_val
		disable_y = True
	if disable_x: xl_access = lambda d: str(d[x_title])
	else: xl_access = lambda d: str(d[last_x_val][xaxis_column_sub_name])
	if disable_y: yl_access = lambda d: str(d[y_title])
	else: yl_access = lambda d: str(d[last_y_val][yaxis_column_sub_name])
	label_data = [
		[
			x_title+": "+xl_access(d)+"<br>"
			+y_title+": "+yl_access(d)
			for d in data
		]
		for data in data_groups
	]
	# Crop data by min and max values
	x_min = float(x_min_val) if len(x_min_val) > 0 else 0
	x_max = float(x_max_val) if len(x_max_val) > 0 else math.inf
	y_min = float(y_min_val) if len(y_min_val) > 0 else 0
	y_max = float(y_max_val) if len(y_max_val) > 0 else math.inf
	for i in range(len(x_data)):
		x_data[i], y_data[i], label_data[i] = cross_filter(
			lambda x,y,z:
				x != "" and y != "" and z != ""
				and float(x) >= x_min
				and float(x) <= x_max
				and float(y) >= y_min
				and float(y) <= y_max,
			[x_data[i],y_data[i],label_data[i]])
		if len(x_data[i]) == 0:
			continue
		x_data[i], y_data[i], label_data[i] = cross_sort(
			[x_data[i],y_data[i],label_data[i]]
		)
	x_data,y_data,label_data,data_groups = cross_filter(lambda xd,yd,ld,gd: len(xd) > 0 and len(yd) > 0, [x_data,y_data,label_data,data_groups])
	# Crop data for fitting
	x_data_fit = x_data
	y_data_fit = y_data
	if str(yaxis_type) == "Log":
		x_data_fit = []
		y_data_fit = []
		for xd,yd in zip(x_data,y_data):
			xrd,yrd = cross_filter(lambda x,y: float(x) > 0 and float(y) > 0, [xd,yd])
			x_data_fit.append([math.log(float(x)) for x in xrd])
			y_data_fit.append([math.log(float(y)) for y in yrd])
	# Comput power fit
	linear_regressor = LinearRegression()  # create object for the class
	regressions = [
		linear_regressor.fit(
			[[a] for a in x],
			[[a] for a in y]
		).coef_[0][0]
		if len(x) > 0 else 0
		for x,y in zip(x_data_fit,y_data_fit)
	]
	
	group_names = [
		', '.join([
			groups[gi]+": "+str(data_groups[i][0][groups[gi]])
			for gi in group_checks
		])
		+ ', ' + yaxis_type + ' regression: '
		+ str(regressions[i])
		for i in range(len(data_groups))
	]

	return {
		'data': [go.Scatter(
			x=x_data[i],
			y=y_data[i],
			text=label_data[i],
			customdata=label_data[i],
			name=group_names[i],
			mode='markers+lines',
			marker={
				'size': 15,
				'opacity': 0.5,
				'line': {'width': 0.5, 'color': 'white'}
			}
		) for i in range(len(data_groups))],
		'layout': go.Layout(
			xaxis={
				'title': x_title,
				'type': 'linear' if xaxis_type == 'Linear' else 'log'
			},
			yaxis={
				'title': y_title,
				'type': 'linear' if yaxis_type == 'Linear' else 'log'
			},
			margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
			height=450,
			hovermode='closest'
		)
	}

if __name__ == '__main__':
    app.run_server(debug=False)
