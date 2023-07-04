import os
from flask import Flask
import flask_uploads


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        UPLOADS_DEFAULT_DEST='innvestigate-gui/static/input'
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from . import main
    app.register_blueprint(main.bp)
    app.add_url_rule('/', endpoint='main')
    
    # configure Flask-Uploads
    flask_uploads.configure_uploads(app, (main.images_UploadSet, main.models_UploadSet, main.class_indexes_UploadSet))

    return app
