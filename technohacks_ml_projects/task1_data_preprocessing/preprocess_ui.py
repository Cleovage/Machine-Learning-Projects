import tkinter as tk
from tkinter import filedialog, messagebox
import os
from preprocess_titanic import preprocess_data

class PreprocessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Titanic Data Preprocessor")

        self.input_filepath = tk.StringVar()
        self.output_filepath = tk.StringVar()

        # Input File Selection
        tk.Label(root, text="Input CSV File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(root, textvariable=self.input_filepath, width=50).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.browse_input_file).grid(row=0, column=2, padx=5, pady=5)

        # Output File Selection
        tk.Label(root, text="Output CSV File:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        tk.Entry(root, textvariable=self.output_filepath, width=50).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.browse_output_file).grid(row=1, column=2, padx=5, pady=5)

        # Process Button
        tk.Button(root, text="Process Data", command=self.process_data).grid(row=2, column=1, padx=5, pady=10)

    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd(), "technohacks_ml_projects"),
            title="Select Input CSV",
            filetypes=(("CSV files", "*.csv"), ("all files", "*.*"))
        )
        if filename:
            self.input_filepath.set(filename)

    def browse_output_file(self):
        filename = filedialog.asksaveasfilename(
            initialdir=os.path.join(os.getcwd(), "technohacks_ml_projects", "task1_data_preprocessing"),
            title="Save Cleaned CSV As",
            filetypes=(("CSV files", "*.csv"), ("all files", "*.*")),
            defaultextension=".csv"
        )
        if filename:
            self.output_filepath.set(filename)

    def process_data(self):
        input_path = self.input_filepath.get()
        output_path = self.output_filepath.get()

        if not input_path:
            messagebox.showerror("Error", "Please select an input CSV file.")
            return
        if not output_path:
            messagebox.showerror("Error", "Please specify an output CSV file.")
            return

        try:
            preprocess_data(input_path, output_path)
            messagebox.showinfo("Success", f"Data preprocessed and saved to {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PreprocessApp(root)
    root.mainloop()
