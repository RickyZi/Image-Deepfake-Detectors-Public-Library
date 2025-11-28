import os
import glob 
import pandas as pd
import numpy as np

def name_mapper(name):
    gen_keys = {
        'gan1':['SG'],
        'gan2':['SG2'],
        'gan3':['SG3'],
        'sd15':['SD1.5'],
        'sd2':['SD2.1'],
        'sd3':['SD3'],
        'sdXL':['SDXL'],
        'flux':['FLUX.1'],
        'realFFHQ':['FFHQ'],
        'realFORLAB':['FORLAB']
    }
    return gen_keys[name][0]

def to_latex_table(df, caption, train_key):
    first_line = df.columns.to_list()

    with open(f'latex/{train_key}.txt', 'a') as f:
        header = f'\\begin{{table*}}[]\n\\caption{{{caption}}}\n\\begin{{tblr}}{{{"l"*len(first_line)}}}\n'
        f.write(header)

        header = f'{first_line[0]} & {" & ".join([name_mapper(first_line[i].split(":")[0]) for i in range(1, len(first_line))])} \\\ \hline \hline \n'
        f.write(header)

        for i, row in df.iterrows():
            # row = ' & '.join([row[0]] + [f'{row[i]:.2f}' for i in range(1, len(row))]) 
            row = f'\SetCell[r=2]{{l}} {row[0]} & '.replace('R50_TF', 'Ours').replace('_', '\_').replace('nodown', 'ND') + " & ".join([f'\SetCell[r=2]{{c}} {row[i]:.2f}' for i in range(1, len(row))]) + ' \\\ \\\ \hline \n'
            f.write(row)

        footer = f'{"&"*(len(first_line)-1)}\n\\end{{tblr}}\n\\label{{tab:dataset}}\n\\end{{table*}}\n\n'
        f.write(footer)
    
def to_latex_table_trans(df, caption, train_key):
    df = df.set_index('Detector')
    df = df.T.reset_index(names=['Dataset'])
    first_line = df.columns.to_list()

    with open(f'latex/{train_key}.txt', 'a') as f:
        header = f'\\begin{{table}}[]\n\\caption{{{caption}}}\n\\begin{{tblr}}{{{"X[1,l,m]"+ "X[1,c,m]"*(len(first_line)-1)}}}\n'
        f.write(header)

        header = f'{" & ".join([first_line[i] for i in range(0, len(first_line))])} \\\ \hline \hline \n'.replace("R50_TF", "Ours").replace("_", "\_").replace("nodown", "ND")
        f.write(header)

        for i, row in df.iterrows():
            # row = ' & '.join([row[0]] + [f'{row[i]:.2f}' for i in range(1, len(row))]) 
            row = f'{name_mapper(row[0].split(":")[0])} & ' + " & ".join([f'{row[i]:.2f}' for i in range(1, len(row))]) + ' \\\ \hline \n'
            f.write(row)

        footer = f'\end{{tblr}}\n\\label{{tab:dataset}}\n\\end{{table}}\n\n'
        f.write(footer)

def to_latex_table_trans_diff(df, df2, caption, train_key):
    df = df.set_index('Detector')
    df2 = df2.set_index('Detector')
    df = df.T.reset_index(names=['Data'])
    df2 = df2.T.reset_index(names=['Data'])
    first_line = df.columns.to_list()

    with open(f'latex/{train_key}.txt', 'a') as f:
        header = f'\\begin{{table}}[]\n\\caption{{{caption}}}\n\\begin{{tblr}}{{{"l"*len(first_line)}}}\n'
        f.write(header)

        header = f'{" & ".join([first_line[i] for i in range(0, len(first_line))])} \\\ \hline \hline \n'.replace("R50_TF", "Ours").replace("_", "\_").replace("nodown", "ND")
        f.write(header)

        for (i, row), (i2, row2) in zip(df.iterrows(), df2.iterrows()):
            # diffs = [f'{row2[i] - row[i]:.2f}' for i in range(1, len(row))]
            diffs = [row2[i] - row[i] for i in range(1, len(row))]
            diffs_str = [f'{diff:.2f}' for diff in diffs]
            print([f'{diff:.2f}' for diff in diffs])
            # row = ' & '.join([row[0]] + [f'{row[i]:.2f}' for i in range(1, len(row))]) 
            # row_head = f'\SetCell[r=2]{{l}} {name_mapper(row[0].split(":")[0])} & '
            row_head = f'\SetCell[r=3]{{l}} {name_mapper(row[0].split(":")[0])} & ' \
            + f'& Facebook {"&"*(len(first_line)-2)} \\\ \n' \
            + f'& Telegram {"&"*(len(first_line)-2)} \\\ \n' \
            + f'& Twitter {"&"*(len(first_line)-2)} \\\ \n'
            row_center = " & ".join([f'\SetCell[r=2]{{c}} \\textcolor{{{"red" if diff < 0 else "blue"}}}{{{diff_str}}}' for diff, diff_str in zip(diffs, diffs_str)])
            row_tail = ' \\\ \\\ \hline \n'
            row = row_head + row_center + row_tail
            f.write(row)

        footer = f'{"&"*(len(first_line)-1)}\n\\end{{tblr}}\n\\label{{tab:dataset}}\n\\end{{table}}\n\n'
        f.write(footer)

def to_latex_table_diffs(df, df_fb, df_tl, df_tw, caption, train_key):
    df = df.set_index('Detector')
    df = df.T.reset_index(names=['Data'])
    
    df_fb = df_fb.set_index('Detector')
    df_fb = df_fb.T.reset_index(names=['Data'])
    
    df_tl = df_tl.set_index('Detector')
    df_tl = df_tl.T.reset_index(names=['Data'])
    
    df_tw = df_tw.set_index('Detector')
    df_tw = df_tw.T.reset_index(names=['Data'])

    first_line = df.columns.to_list()

    with open(f'latex/{train_key}.txt', 'a') as f:
        header = f'\\begin{{table}}[]\n\\caption{{{caption}}}\n\\begin{{tblr}}{{rows = {{abovesep=1pt,belowsep=1pt}}, colsep = 2pt, colspec = {{{"X[1,l,m]X[-1,l,m]"+"X[1,c,m]"*(len(first_line)-1)}}}}}\n'

        
        f.write(header)

        header = f'\SetCell[c=2]{{c}} {first_line[0]}' + ' & & '+ f'{" & ".join([first_line[i] for i in range(1, len(first_line))])} \\\ \hline \hline \n'.replace("R50_TF", "Ours").replace("R50_nodown", "R50-ND")
        f.write(header)

        for (i, row), (_, row_fb), (_, row_tl), (_, row_tw)in zip(df.iterrows(), df_fb.iterrows(), df_tl.iterrows(), df_tw.iterrows()):
            row_head = f'\SetCell[r=3]{{r}} \\footnotesize {name_mapper(row[0].split(":")[0])} & '
            row_pad = f' & '

            row = [row[0]] + [float(f'{row[i]:.2f}') for i in range(1, len(row))]
            row_fb = [row_fb[0]] + [float(f'{row_fb[i]:.2f}') for i in range(1, len(row_fb))]
            row_tl = [row_tl[0]] + [float(f'{row_tl[i]:.2f}') for i in range(1, len(row_tl))]
            row_tw = [row_tw[0]] + [float(f'{row_tw[i]:.2f}') for i in range(1, len(row_tw))]

            row_center_fb = "\\tiny FB & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black" if abs(row_fb[i] - row[i]) < 0.05 else ("red" if row_fb[i] - row[i] < 0 else "blue")}}}{{{row_fb[i]:.2f} \\tiny {row_fb[i] - row[i]:+.2f}}}' for i in range(1, len(row))])
            row_center_tl = "\\tiny TL & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black" if abs(row_tl[i] - row[i]) < 0.05 else ("red" if row_tl[i] - row[i] < 0 else "blue")}}}{{{row_tl[i]:.2f} \\tiny {row_tl[i] - row[i]:+.2f}}}' for i in range(1, len(row))])
            row_center_tw = "\\tiny TW & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black" if abs(row_tw[i] - row[i]) < 0.05 else ("red" if row_tw[i] - row[i] < 0 else "blue")}}}{{{row_tw[i]:.2f} \\tiny {row_tw[i] - row[i]:+.2f}}}' for i in range(1, len(row))])

            row_tail_middle = ' \\\ \n'
            row_tail_end = ' \\\ \hline \n'
            row = row_head + row_center_fb + row_tail_middle + row_pad + row_center_tl + row_tail_middle + row_pad + row_center_tw + row_tail_end
            f.write(row)

        footer = f'\\end{{tblr}}\n\\label{{tab:dataset}}\n\\end{{table}}\n\n'
        f.write(footer)

def to_latex_table_all_trans(df, df_fb, df_tl, df_tw, caption, train_key):
    df = df.set_index('Detector')
    df = df.T.reset_index(names=['Data'])
    
    df_fb = df_fb.set_index('Detector')
    df_fb = df_fb.T.reset_index(names=['Data'])
    
    df_tl = df_tl.set_index('Detector')
    df_tl = df_tl.T.reset_index(names=['Data'])
    
    df_tw = df_tw.set_index('Detector')
    df_tw = df_tw.T.reset_index(names=['Data'])

    first_line = df.columns.to_list()


    with open(f'latex/{train_key}.txt', 'a') as f:
        header = f'\\begin{{table}}[]\n\\caption{{{caption}}}\n\\begin{{tblr}}{{rows = {{abovesep=1pt,belowsep=1pt}}, colsep = 2pt, colspec = {{{"X[1,l,m]X[-1,l,m]"+"X[1,c,m]"*(len(first_line)-1)}}}}}\n'

        
        f.write(header)

        header = f'\SetCell[c=2]{{c}} {first_line[0]}' + ' & & '+ f'{" & ".join([first_line[i] for i in range(1, len(first_line))])} \\\ \hline \hline \n'.replace("R50_TF", "Ours").replace("R50_nodown", "R50-ND")
        f.write(header)

        for (i, row), (_, row_fb), (_, row_tl), (_, row_tw)in zip(df.iterrows(), df_fb.iterrows(), df_tl.iterrows(), df_tw.iterrows()):
            row_head = f'\SetCell[r=4]{{r}} \\footnotesize {name_mapper(row[0].split(":")[0])} & '
            row_pad = f' & '

            row = [row[0]] + [float(f'{row[i]:.2f}') for i in range(1, len(row))]
            row_fb = [row_fb[0]] + [float(f'{row_fb[i]:.2f}') for i in range(1, len(row_fb))]
            row_tl = [row_tl[0]] + [float(f'{row_tl[i]:.2f}') for i in range(1, len(row_tl))]
            row_tw = [row_tw[0]] + [float(f'{row_tw[i]:.2f}') for i in range(1, len(row_tw))]

            row_center = "\\scriptsize \\textbf{Pre} & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black"}}}{{{row[i]:.2f}}}' for i in range(1, len(row))])
            row_center_fb = "\\scriptsize FB & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black" if abs(row_fb[i] - row[i]) < 0.01 else ("red" if row_fb[i] - row[i] < 0 else "blue")}}}{{{row_fb[i]:.2f} \\tiny {row_fb[i] - row[i]:+.2f}}}' for i in range(1, len(row))])
            row_center_tl = "\\scriptsize TL & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black" if abs(row_tl[i] - row[i]) < 0.01 else ("red" if row_tl[i] - row[i] < 0 else "blue")}}}{{{row_tl[i]:.2f} \\tiny {row_tl[i] - row[i]:+.2f}}}' for i in range(1, len(row))])
            row_center_tw = "\\scriptsize TW & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black" if abs(row_tw[i] - row[i]) < 0.01 else ("red" if row_tw[i] - row[i] < 0 else "blue")}}}{{{row_tw[i]:.2f} \\tiny {row_tw[i] - row[i]:+.2f}}}' for i in range(1, len(row))])

            row_tail_middle = ' \\\ \n'
            row_tail_end = ' \\\ \hline \n'
            row = row_head + row_center + row_tail_middle + row_pad +  row_center_fb + row_tail_middle + row_pad + row_center_tl + row_tail_middle + row_pad + row_center_tw + row_tail_end
            f.write(row)

        footer = f'\\end{{tblr}}\n\\label{{tab:dataset}}\n\\end{{table}}\n\n'
        f.write(footer)
    
def to_latex_table_all_trans2(df, df_fb, df_tl, df_tw, caption, train_key):
    df = df.set_index('Detector')
    df = df.T.reset_index(names=['Data'])
    
    df_fb = df_fb.set_index('Detector')
    df_fb = df_fb.T.reset_index(names=['Data'])
    
    df_tl = df_tl.set_index('Detector')
    df_tl = df_tl.T.reset_index(names=['Data'])
    
    df_tw = df_tw.set_index('Detector')
    df_tw = df_tw.T.reset_index(names=['Data'])

    first_line = df.columns.to_list()


    with open(f'latex/{train_key}.txt', 'a') as f:
        header = f'\\begin{{table}}[]\n\\caption{{{caption}}}\n\\begin{{tblr}}{{rows = {{abovesep=1pt,belowsep=1pt}}, colsep = 2pt, colspec = {{{"X[1,l,m]"+"X[1,r,m]X[1,l,m]"*(len(first_line)-1)}}}}}\n'
        f.write(header)

        header = f'{first_line[0]} & \SetCell[c=2]{{c}} ' + f'{" && TMP ".join([first_line[i] for i in range(1, len(first_line))])} \\\ \hline \hline \n'.replace("R50_TF", "Ours").replace("R50_nodown", "R50-ND").replace('TMP', '\SetCell[c=2]{c}')
        f.write(header)

        for (i, row), (_, row_fb), (_, row_tl), (_, row_tw)in zip(df.iterrows(), df_fb.iterrows(), df_tl.iterrows(), df_tw.iterrows()):
          
            row = [row[0]] + [float(f'{row[i]:.2f}') for i in range(1, len(row))]
            row_fb = [row_fb[0]] + [float(f'{row_fb[i]:.2f}') for i in range(1, len(row_fb))]
            row_tl = [row_tl[0]] + [float(f'{row_tl[i]:.2f}') for i in range(1, len(row_tl))]
            row_tw = [row_tw[0]] + [float(f'{row_tw[i]:.2f}') for i in range(1, len(row_tw))]  

            first_row = f'\SetCell[r=3]{{l}} \\footnotesize {name_mapper(row[0].split(":")[0])} & ' + ' & '.join([f'\SetCell[r=3]{{r}} \\textcolor{{{"black"}}}{{{row[i]:.2f}}} & ' + f'\\tiny FB \\scriptsize \\textcolor{{{"black" if abs(row_fb[i] - row[i]) < 0.01 else ("red" if row_fb[i] - row[i] < 0 else "blue")}}}{{{row_fb[i]:.2f}}}' for i in range(1, len(row))]) + ' \\\ \n'
            # first_row = f'\SetCell[r=3]{{r}} \\footnotesize {name_mapper(row[0].split(":")[0])} & ' + ' & '.join([f'\SetCell[r=3]{{r}} \\textcolor{{{"black"}}}{{{row[i]:.2f}}} & ' + f'\\textcolor{{{"black" if abs(row_fb[i] - row[i]) < 0.01 else ("red" if row_fb[i] - row[i] < 0 else "blue")}}}{{{row_fb[i]:.2f} \\tiny {row_fb[i] - row[i]:+.2f}}}' for i in range(1, len(row))]) + ' \\\ \n'
            second_row = ' & ' + ' & '.join([' & ' + f'\\tiny TL \\scriptsize \\textcolor{{{"black" if abs(row_tl[i] - row[i]) < 0.01 else ("red" if row_tl[i] - row[i] < 0 else "blue")}}}{{{row_tl[i]:.2f}}}' for i in range(1, len(row))]) + ' \\\ \n'
            # second_row = ' & ' + ' & '.join([' & ' + f'\\textcolor{{{"black" if abs(row_tl[i] - row[i]) < 0.01 else ("red" if row_tl[i] - row[i] < 0 else "blue")}}}{{{row_tl[i]:.2f} \\tiny {row_tl[i] - row[i]:+.2f}}}' for i in range(1, len(row))]) + ' \\\ \n'
            third_row = ' & ' + ' & '.join([' & ' + f'\\tiny TW  \\scriptsize\\textcolor{{{"black" if abs(row_tw[i] - row[i]) < 0.01 else ("red" if row_tw[i] - row[i] < 0 else "blue")}}}{{{row_tw[i]:.2f}}}' for i in range(1, len(row))]) + ' \\\ \hline \n'
            # third_row = ' & ' + ' & '.join([' & ' + f'\\textcolor{{{"black" if abs(row_tw[i] - row[i]) < 0.01 else ("red" if row_tw[i] - row[i] < 0 else "blue")}}}{{{row_tw[i]:.2f} \\tiny {row_tw[i] - row[i]:+.2f}}}' for i in range(1, len(row))]) + ' \\\ \hline \n'
            row = first_row + second_row + third_row
            f.write(row)

        footer = f'\\end{{tblr}}\n\\label{{tab:dataset}}\n\\end{{table}}\n\n'
        f.write(footer)

def to_latex_table_all(df, df_fb, df_tl, df_tw, caption, train_key):
    first_line = df.columns.to_list()


    with open(f'latex/{train_key}.txt', 'a') as f:
        header = f'\\begin{{table*}}[]\n\\caption{{{caption}}}\n\\begin{{tblr}}{{rows = {{abovesep=1pt,belowsep=1pt}}, colsep = 2pt, colspec = {{{"X[1,l,m]X[1,l,m]"+"X[1,c,m]"*(len(first_line)-1)}}}}}\n'

        
        f.write(header)
        header = f'{first_line[0]} & & ' + " & ".join([f'{name_mapper(first_line[i].split(":")[0])} ' for i in range(1, len(first_line))]) + ' \\\ \hline \hline \n'
        
        #header = f'{first_line[0]}' + ' & & '+ f'{" & ".join([first_line[i] for i in range(1, len(first_line))])} \\\ \hline \hline \n'.replace("R50_TF", "Ours").replace("_", "\_").replace("nodown", "ND")
        f.write(header)

        for (i, row), (_, row_fb), (_, row_tl), (_, row_tw)in zip(df.iterrows(), df_fb.iterrows(), df_tl.iterrows(), df_tw.iterrows()):
            row_head = f'\SetCell[r=4]{{r}} \\footnotesize ' + f'{row[0]} & '.replace('R50_TF', 'Ours').replace('R50_nodown', 'R50-ND')
            row_pad = f' & '


            row = [row[0]] + [float(f'{row[i]:.2f}') for i in range(1, len(row))]
            row_fb = [row_fb[0]] + [float(f'{row_fb[i]:.2f}') for i in range(1, len(row_fb))]
            row_tl = [row_tl[0]] + [float(f'{row_tl[i]:.2f}') for i in range(1, len(row_tl))]
            row_tw = [row_tw[0]] + [float(f'{row_tw[i]:.2f}') for i in range(1, len(row_tw))]

            # row_center = "\\tiny \\textbf{PreSocial} & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black"}}}{{{row[i]:.2f}}}' for i in range(1, len(row))])
            # row_center_fb = "\\tiny FB & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black" if abs(row_fb[i] - row[i]) < 0.01 else ("red" if row_fb[i] - row[i] < 0 else "blue")}}}{{{row_fb[i]:.2f} \\tiny {row_fb[i] - row[i]:+.2f}}}' for i in range(1, len(row))])
            # row_center_tl = "\\tiny TL & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black" if abs(row_tl[i] - row[i]) < 0.01 else ("red" if row_tl[i] - row[i] < 0 else "blue")}}}{{{row_tl[i]:.2f} \\tiny {row_tl[i] - row[i]:+.2f}}}' for i in range(1, len(row))])
            # row_center_tw = "\\tiny TW & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black" if abs(row_tw[i] - row[i]) < 0.01 else ("red" if row_tw[i] - row[i] < 0 else "blue")}}}{{{row_tw[i]:.2f} \\tiny {row_tw[i] - row[i]:+.2f}}}' for i in range(1, len(row))])

            row_tail_middle = ' \\\ \n'
            row_tail_end = ' \\\ \hline \n'
            # row = row_head + row_center + row_tail_middle + row_pad +  row_center_fb + row_tail_middle + row_pad + row_center_tl + row_tail_middle + row_pad + row_center_tw + row_tail_end

            row_center      = "\\tiny Pre & \\scriptsize " + " & \\scriptsize ".join([f'{row[i]:.2f}' for i in range(1, len(row))])
            row_center_fb   = "\\tiny FB & \\scriptsize " + " & \\scriptsize ".join([f'{row_fb[i]:.2f}' for i in range(1, len(row))])
            row_center_tl   = "\\tiny TL & \\scriptsize " + " & \\scriptsize ".join([f'{row_tl[i]:.2f}' for i in range(1, len(row))])
            row_center_tw   = "\\tiny TW & \\scriptsize " + " & \\scriptsize ".join([f'{row_tw[i]:.2f}' for i in range(1, len(row))])

            # row_center      = "\\tiny Pre & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black"}}}{{{row[i]:.2f}}}' for i in range(1, len(row))])
            # row_center_fb   = "\\tiny FB & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black" if abs(row_fb[i] - row[i]) < 0.01 else ("red" if row_fb[i] - row[i] < 0 else "blue")}}}{{{row_fb[i]:.2f} \\tiny {row_fb[i] - row[i]:+.2f}}}' for i in range(1, len(row))])
            # row_center_tl   = "\\tiny TL & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black" if abs(row_tl[i] - row[i]) < 0.01 else ("red" if row_tl[i] - row[i] < 0 else "blue")}}}{{{row_tl[i]:.2f} \\tiny {row_tl[i] - row[i]:+.2f}}}' for i in range(1, len(row))])
            # row_center_tw   = "\\tiny TW & \\scriptsize " + " & \\scriptsize ".join([f'\\textcolor{{{"black" if abs(row_tw[i] - row[i]) < 0.01 else ("red" if row_tw[i] - row[i] < 0 else "blue")}}}{{{row_tw[i]:.2f} \\tiny {row_tw[i] - row[i]:+.2f}}}' for i in range(1, len(row))])
            
            row = row_head  + row_center + row_tail_middle + row_pad +  row_center_fb + row_tail_middle + row_pad + row_center_tl + row_tail_middle + row_pad + row_center_tw + row_tail_end
            f.write(row)

        footer = f'\\end{{tblr}}\n\\label{{tab:dataset}}\n\\end{{table*}}\n\n'
        f.write(footer)

def column_sorter(list_of_columns):
    columns_real = sorted([column for column in list_of_columns if 'real' in column])
    columns_gan = sorted([column for column in list_of_columns if 'gan' in column])
    columns_sd = sorted([column for column in list_of_columns if 'sd' in column])
    columns_flux = sorted([column for column in list_of_columns if 'flux' in column])
    return columns_real + columns_gan + columns_sd + columns_flux

detectors = ['CLIP-D', 'MISLNet', 'NPR', 'P2G', 'R50_nodown', 'R50_TF']
detectors = ['R50_TF', 'R50_nodown', 'MISLNet', 'NPR', 'CLIP-D', 'P2G']
# detectors = ['R50_TF', 'R50_nodown', 'CLIP-D', 'P2G']

train = {}


for detector in detectors:
    runs = os.listdir(os.path.join('..', 'detectors', detector, 'train'))
    for run in runs:
        train_set = run
        if train_set not in train:
            train[train_set] = {}

        results = glob.glob(os.path.join('..', 'detectors', detector, 'train', run, 'data', '**', '*.csv'))
        for result in results:
            test_set = result.split('/')[-2]
            df = pd.read_csv(result)

            try:
                y_pred = df['pro'].values
            except KeyError:
                y_pred = df['pro_mix'].values
            
            y_pred = y_pred > 0
            
            y_true = df['flag'].values

            recall = np.sum(y_true == y_pred) / len(y_true)

            if test_set not in train[train_set]:
                train[train_set][test_set] = [{'detector': detector, 'result': recall}]
            else:
                train[train_set][test_set].append({'detector': detector, 'result': recall})


for train_key, train_dicts in train.items():
    df_train = pd.DataFrame(columns=['Detector', *train_dicts.keys()])

    test_dict = {}
    for test_key, test_dicts in train_dicts.items():
        detectors = [test_dict['detector'] for test_dict in test_dicts]
        results = [test_dict['result'] for test_dict in test_dicts]
        test_dict[test_key] = dict(zip(detectors, results))

    for detector in detectors:
        scores = [test_dict[test_key][detector] for test_key in test_dict.keys()]
        df_train = df_train._append({'Detector': detector, **dict(zip(list(train_dicts.keys()), scores))}, ignore_index=True)

    try:
        os.remove(f'latex/{train_key}.txt')
        os.remove(f'latex/diff_{train_key}.txt')
    except FileNotFoundError:
        pass
    df_train_pre = df_train[['Detector'] + column_sorter([column for column in df_train.columns if 'pre' in column])]

    df_train_fb = df_train[['Detector'] + column_sorter([column for column in df_train.columns if 'fb' in column])]
    # to_latex_table_trans(df_train_fb, f'Facebook trained on  {train_key}', train_key)
    # to_latex_table_trans_diff(df_train_pre, df_train_fb, f'Delta Pre-FB trained on {train_key}'.replace('&', '\&'), train_key)
    # quit()

    df_train_tl = df_train[['Detector'] + column_sorter([column for column in df_train.columns if 'tl' in column])]
    # to_latex_table_trans(df_train_tl, f'Telegram trained on {train_key}', train_key)
    # to_latex_table_trans_diff(df_train_pre, df_train_tl, f'Delta Pre-TL trained on {train_key}'.replace('&', '\&'), train_key)

    df_train_tw = df_train[['Detector'] + column_sorter([column for column in df_train.columns if 'tw' in column])]
    # to_latex_table_trans(df_train_tw, f'Twitter trained on {train_key}', train_key)
    # to_latex_table_trans_diff(df_train_pre, df_train_tw, f'Delta Pre-TW trained on {train_key}'.replace('&', '\&'), train_key)

    if 'gan' in train_key and 'sd' in train_key:
        to_latex_table_trans(df_train_pre, f'PreSocial, trained on {train_key}'.replace('&', '\&'), train_key)
        to_latex_table_diffs(df_train_pre, df_train_fb, df_train_tl, df_train_tw, f'Deltas, trained on {train_key}'.replace('&', '\&'), train_key)
    
    else:
        to_latex_table_all_trans2(df_train_pre, df_train_fb, df_train_tl, df_train_tw, f'All, trained on {train_key}'.replace('&', '\&'), train_key)
    
    if 'freeze' in train_key or True:
        df_train_pre.to_csv(f'results/{train_key}_pre.csv', index=False, float_format='%.2f')
        df_train_fb.to_csv(f'results/{train_key}_fb.csv', index=False, float_format='%.2f')
        df_train_tl.to_csv(f'results/{train_key}_tl.csv', index=False, float_format='%.2f')
        df_train_tw.to_csv(f'results/{train_key}_tw.csv', index=False, float_format='%.2f')

        lines = []
        for csv in ['pre', 'fb', 'tl', 'tw']:
            with open(f'results/{train_key}_{csv}.csv', 'r') as f:
                lines.extend(f.readlines())
            lines.append('\n')

        with open(f'results/{train_key}.csv', 'w') as f:
            f.writelines(lines)


