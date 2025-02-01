from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from Classification import classify_image

app = Flask(__name__)
CORS(app)


@app.route('/home', methods=['POST'])
def process_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        uploaded_file = request.files['file']
        print("Received file:", uploaded_file.filename)
        print("Content type:", uploaded_file.content_type)

        image = Image.open(uploaded_file)
        width, height = image.size

        results = {"classification": "test"}
        results['results'] = classify_image(image)
        print(results['results'])

        return jsonify({
            "message": "Image processed successfully",
            "width": width,
            "height": height,
            "results": results
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
