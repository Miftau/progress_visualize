from flask import Flask, request, render_template, jsonify
from visualize import load_data, calculate_statistics, generate_visualizations

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    if request.method == 'POST':
        file = request.files.get('file')
        raw_data = request.form.get('raw_data')

        print("Received file:", file.filename if file else "No file")  # Debug
        print("Received raw_data:", raw_data if raw_data else "No raw_data")  # Debug

        if not file and not raw_data:
            error = "Please upload a file or provide raw data."
            print(error)  # Debug
            return render_template('index.html', error=error)

        try:
            df = load_data(file, raw_data)
            print("Data loaded successfully:", df.head())  # Debug
            stats_table = calculate_statistics(df)
            print("Statistics calculated:", stats_table)  # Debug
            plots = generate_visualizations(df)
            print("Plots generated:", len(plots), "plots")  # Debug
            try:
                return render_template('result.html', stats=stats_table.to_html(classes='table table-striped'), plots=plots)
            except Exception as render_error:
                error = f"Error rendering template: {str(render_error)}"
                print(error)  # Debug
                return render_template('index.html', error=error)
        except Exception as e:
            error = f"Error processing data: {str(e)}"
            print(error)  # Debug
            return render_template('index.html', error=error)

    return render_template('index.html', error=error)

if __name__ == '__main__':
    app.run(debug=True)