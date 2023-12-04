# Define functions for plotting depth (step) data as a time series (with gap checking) or
# multivariable correlation matrix. Functions are caled in a separate script (no data import features here)

import pandas as pd
import numpy as np
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime, timedelta
import mplcursors
import seaborn as sns
import matplotlib.pyplot as plt


def step_time_series_plot(dataframe, title):
    """Create a scalable plot for viewing multiparameter time series dataset"""
    def create_scatter_plot(variables, start_date, end_date, draw_lines=False):
        fig.clf()  # Clear the current figure to update it

        ax = fig.add_subplot(111)
        legend_labels = []
        scatter_list = []  # List to store scatter plots
        plot_list = []  # List to store scatter plots

        for variable in variables:
            if variable_visibility[variable].get() == 1:

                data = dataframe[(dataframe["Date"] >= start_date) & (dataframe["Date"] <= end_date)]
                scatter = ax.scatter(data["Date"], data[variable], label=variable, s=5, alpha=0.5)
                scatter_list.append(scatter)  # Append scatter plot to the list
                if draw_lines:
                    plot_line, = ax.plot(data["Date"], data[variable], marker='None',
                                         linestyle='-', label="_nolegend_")
                    plot_list.append(plot_line)  # Append plot line to the list

                legend_labels.append(variable)

                # Add tooltips using mplcursors
                cursor = mplcursors.cursor(hover=True)
                cursor.connect("add", lambda sel: display_tooltip(sel, scatter_list))

        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.grid(True)

        if legend_labels:
            ax.legend(legend_labels, loc="upper right")

        canvas.draw()

    def on_checkbox_change():
        start_date = datetime.fromtimestamp(start_slider.get())
        end_date = datetime.fromtimestamp(end_slider.get())
        create_scatter_plot(variables, start_date, end_date, draw_lines=draw_lines_var.get())
        update_slider_labels()  # Update slider labels

    def on_select_all():
        for var in variable_visibility.values():
            var.set(select_all_var.get())
        on_checkbox_change()  # Update the plot

    def on_start_slider_change(event):
        start_date = datetime.fromtimestamp(start_slider.get())
        end_date = datetime.fromtimestamp(end_slider.get())
        create_scatter_plot(variables, start_date, end_date, draw_lines=draw_lines_var.get())
        update_slider_labels()  # Update slider labels

    def on_end_slider_change(event):
        start_date = datetime.fromtimestamp(start_slider.get())
        end_date = datetime.fromtimestamp(end_slider.get())
        create_scatter_plot(variables, start_date, end_date, draw_lines=draw_lines_var.get())
        update_slider_labels()  # Update slider labels

    def update_slider_labels():
        # Update slider labels with datetime format
        start_label_var.set(datetime.fromtimestamp(start_slider.get()).strftime("%Y-%m-%d %H:%M:%S"))
        end_label_var.set(datetime.fromtimestamp(end_slider.get()).strftime("%Y-%m-%d %H:%M:%S"))

    def on_draw_lines_change():
        start_date = datetime.fromtimestamp(start_slider.get())
        end_date = datetime.fromtimestamp(end_slider.get())
        create_scatter_plot(variables, start_date, end_date, draw_lines=draw_lines_var.get())
        update_slider_labels()  # Update slider labels

    def insert_gaps(df, threshold):
        df_with_gaps = df.copy()
        new_rows = []

        for i in range(1, len(df)):
            interval = (df.index[i] - df.index[i - 1]).days  # Assuming datetime objects

            if interval > threshold:
                new_index = df.index[i] - timedelta(days=threshold)
                new_row = pd.DataFrame(index=[new_index], columns=df.columns, data=np.nan)
                new_rows.append(new_row)

        if new_rows:
            new_df = pd.concat([df_with_gaps] + new_rows)
            df_with_gaps = new_df.sort_index()
        df_with_gaps = df_with_gaps.rename_axis('Date').reset_index()
        return df_with_gaps

    def display_tooltip(sel, scatter_list):
        x_value = sel.artist.get_offsets()[sel.target.index][0]
        y_value = sel.artist.get_offsets()[sel.target.index][1]

        for scatter in scatter_list:
            if sel.artist in scatter.collections:
                variable_name = scatter.get_label()

                tooltip_text = f"{variable_name}\nX: {x_value}\nY: {y_value}"
                sel.annotation.set_text(tooltip_text)
                break

    root = Tk()
    root.title("Parameters")

    # Set a threshold interval for gaps
    threshold_interval = 1.5  # Set the maximum interval between records for which continuous plotting is allowed

    # Insert placeholders for gaps (np.nan) based on the threshold to prevent plotting lines through gaps in data
    # dataframe = insert_gaps(dataframe, threshold_interval)

    # Create a Figure and set it up for embedding in tkinter
    fig = Figure(figsize=(12, 6), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    variables = dataframe.columns[5:]  # Exclude 'Date', 'Time', 'Latitude', 'Longitude',
    variable_visibility = {var: IntVar() for var in variables}

    start_date = dataframe["Date"].min()
    end_date = dataframe["Date"].max()
    print('start/end', start_date, end_date)

    # Convert the dates to timestamps (float values)
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.min.time())

    start_date_timestamp = start_date.timestamp()
    end_date_timestamp = end_date.timestamp()

    # Create a frame for the sliders
    slider_frame = Frame(root)
    slider_frame.pack(fill=X)

    start_slider = Scale(slider_frame, from_=start_date_timestamp, to=end_date_timestamp, orient=HORIZONTAL,
                         label="Start Date", command=on_start_slider_change)
    start_slider.pack(fill=X, padx=10)
    start_slider.set(start_date_timestamp)  # Set the initial value

    end_slider = Scale(slider_frame, from_=start_date_timestamp, to=end_date_timestamp, orient=HORIZONTAL,
                       label="End Date", command=on_end_slider_change)
    end_slider.pack(fill=X, padx=10)
    end_slider.set(end_date_timestamp)  # Set the initial value

    # Labels for the slider values
    start_label_var = StringVar()
    end_label_var = StringVar()

    start_label = Label(slider_frame, textvariable=start_label_var)
    start_label.pack(side=LEFT, fill=X, padx=10)

    end_label = Label(slider_frame, textvariable=end_label_var)
    end_label.pack(side=LEFT, fill=X, padx=10)

    # Add a label to the checkboxes outside the frame
    title_label = Label(root, text="Parameters", font=("Helvetica", 14, "bold"))
    title_label.pack(pady=5)

    # Create a frame for parameter checkboxes arranged in two columns
    checkbox_frame = Frame(root, bd=2, relief="groove")  # Add border to the checkbox frame
    checkbox_frame.pack()

    left_column = len(variables) // 2
    right_column = len(variables) - left_column

    select_all_var = IntVar()

    select_all_checkbox = Checkbutton(checkbox_frame, text="Select All", variable=select_all_var, command=on_select_all)
    select_all_checkbox.grid(row=0, column=0, sticky="w", columnspan=2)

    # Create checkboxes for the left column
    for i in range(left_column):
        variable = variables[i]
        variable_checkbutton = Checkbutton(checkbox_frame, text=variable, variable=variable_visibility[variable],
                                           command=on_checkbox_change)
        variable_checkbutton.grid(row=i + 1, column=0, sticky="w")

    # Create checkboxes for the right column
    for i in range(right_column):
        variable = variables[left_column + i]
        variable_checkbutton = Checkbutton(checkbox_frame, text=variable, variable=variable_visibility[variable],
                                           command=on_checkbox_change)
        variable_checkbutton.grid(row=i, column=1, sticky="w")

    # Create a separate frame for the "Draw Lines" checkbox
    draw_lines_frame = Frame(root)
    draw_lines_frame.pack()

    draw_lines_var = BooleanVar()
    draw_lines_checkbox = Checkbutton(draw_lines_frame, text="Draw Lines", variable=draw_lines_var,
                                      command=on_draw_lines_change)
    draw_lines_checkbox.pack()

    create_scatter_plot(variables, start_date, end_date)
    update_slider_labels()  # Initialize slider labels

    root.mainloop()


def step_correlate_matrix(df, title):
    """Plot a correlation matrix for all variables in a dataframe"""

    # Remove index variable
    df = df.drop('Record Number', axis=1)
    correlation_matrix = df.corr()

    # Set the figure size and create a subplot with adjusted spacing
    fig, heatmap_ax = plt.subplots(figsize=(12, 10), gridspec_kw={'hspace': 0.5, 'wspace': 0.5})

    # Adjust the size of the heatmap within the subplot
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                xticklabels=df.columns, yticklabels=df.columns,
                cbar_kws={'label': 'Correlation', 'shrink': 0.8}, ax=heatmap_ax)

    # Rotate the column names for better readability
    heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), rotation=45, ha='right')

    plt.title(title)

    # Rotate the column names for better readability
    plt.xticks(rotation=45, ha='right')
    plt.show()
