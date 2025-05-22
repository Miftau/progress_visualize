from flask import Flask, request, render_template, jsonify
from jinja2 import TemplateNotFound
from visualize import load_data, calculate_statistics, generate_visualizations

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    if request.method == 'POST':
        file = request.files.get('file')
        raw_data = request.form.get('raw_data')
        sample_size = int(request.form.get('sample_size', 100000))
        plot_width = int(request.form.get('plot_width', 800))
        plot_height = int(request.form.get('plot_height', 600))

        print("Received file:", file.filename if file else "No file")
        print("Received raw_data:", raw_data if raw_data else "No raw_data")
        print(f"Sample size: {sample_size}, Plot width: {plot_width}, Plot height: {plot_height}")

        if not file and not raw_data:
            error = "Please upload a file or provide raw data."
            print(error)
            return render_template('index.html', error=error)

        try:
            df = load_data(file, raw_data, sample_size=sample_size)
            print("Data loaded successfully:", df.head())
            stats_table = calculate_statistics(df)
            print("Statistics calculated:", stats_table)
            plots = generate_visualizations(df, width=plot_width, height=plot_height)
            print("Plots generated:", len(plots), "plots")
            try:
                return render_template('result.html', stats=stats_table.to_html(classes='table table-striped'), plots=plots)
            except TemplateNotFound as tnf:
                error = f"Template not found: {str(tnf)}"
                print(error)
                return render_template('index.html', error=error)
            except Exception as render_error:
                error = f"Error rendering template: {str(render_error)}"
                print(error)
                return render_template('index.html', error=error)
        except Exception as e:
            error = f"Error processing data: {str(e)}"
            print(error)
            return render_template('index.html', error=error)

    return render_template('index.html', error=error)

if __name__ == '__main__':
    app.run(debug=True)