PS C:\PROJECTS\Paint by numbers> python app.py
 * Serving Flask app 'app'
 * Debug mode: on
2025-03-22 22:47:36,781 - werkzeug - INFO - WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
2025-03-22 22:47:36,781 - werkzeug - INFO - Press CTRL+C to quit
2025-03-22 22:47:36,783 - werkzeug - INFO -  * Restarting with stat
2025-03-22 22:47:41,455 - werkzeug - WARNING -  * Debugger is active!
2025-03-22 22:47:41,471 - werkzeug - INFO -  * Debugger PIN: 899-805-914
2025-03-22 22:47:41,530 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:47:41] "GET / HTTP/1.1" 302 -
2025-03-22 22:47:41,537 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:47:41] "GET /enhanced/ HTTP/1.1" 200 -
2025-03-22 22:47:41,581 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:47:41] "GET /static/css/enhanced.css HTTP/1.1" 304 -
2025-03-22 22:47:41,586 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:47:41] "GET /static/css/style.css HTTP/1.1" 304 -
2025-03-22 22:47:47,211 - pbn-app - INFO - Auto-detected image type: still_life (confidence: 0.48)
2025-03-22 22:47:47,212 - pbn-app - INFO - Low confidence, using general image type
2025-03-22 22:47:47,213 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:47:47] "POST /enhanced/upload HTTP/1.1" 200 -
2025-03-22 22:47:47,220 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:47:47] "GET /enhanced/editor HTTP/1.1" 200 -
2025-03-22 22:47:47,228 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:47:47] "GET /static/css/style.css HTTP/1.1" 304 -
2025-03-22 22:47:47,235 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:47:47] "GET /static/css/enhanced.css HTTP/1.1" 304 -
2025-03-22 22:47:47,236 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:47:47] "GET /static/js/enhanced-editor.js HTTP/1.1" 304 -
2025-03-22 22:47:47,245 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:47:47] "GET /enhanced/api/parameters HTTP/1.1" 200 -
2025-03-22 22:47:47,257 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:47:47] "GET /uploads/1742676467_FELV-cat.jpg HTTP/1.1" 500 -
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 917, in full_dispatch_request
    rv = self.dispatch_request()
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 891, in dispatch_request
    self.raise_routing_exception(req)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 500, in raise_routing_exception
    raise request.routing_exception  # type: ignore[misc]
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\ctx.py", line 362, in match_request
    result = self.url_adapter.match(return_rule=True)  # type: ignore
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\werkzeug\routing\map.py", line 629, in match
    raise NotFound() from None
werkzeug.exceptions.NotFound: 404 Not Found: The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1536, in __call__
    return self.wsgi_app(environ, start_response)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1514, in wsgi_app
    response = self.handle_exception(e)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1511, in wsgi_app
    response = self.full_dispatch_request()
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 919, in full_dispatch_request
    rv = self.handle_user_exception(e)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 802, in handle_user_exception
    return self.handle_http_exception(e)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 777, in handle_http_exception
    return self.ensure_sync(handler)(e)  # type: ignore[no-any-return]
  File "C:\PROJECTS\Paint by numbers\app.py", line 35, in page_not_found
    return render_template('error.html', error_code=404, error_message="Page not found"), 404
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\templating.py", line 149, in render_template
    template = app.jinja_env.get_or_select_template(template_name_or_list)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\environment.py", line 1087, in get_or_select_template
    return self.get_template(template_name_or_list, parent, globals)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\environment.py", line 1016, in get_template
    return self._load_template(name, globals)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\environment.py", line 975, in _load_template
    template = self.loader.load(self, name, self.make_globals(globals))
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\loaders.py", line 126, in load
    source, filename, uptodate = self.get_source(environment, name)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\templating.py", line 65, in get_source
    return self._get_source_fast(environment, template)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\templating.py", line 99, in _get_source_fast
    raise TemplateNotFound(template)
jinja2.exceptions.TemplateNotFound: error.html
2025-03-22 22:48:20,526 - pbn-app - INFO - Auto-detected image type: still_life (confidence: 0.53)
2025-03-22 22:48:20,526 - pbn-app.settings_manager - INFO - Applied custom parameters: {'colors': 50, 'edge_strength': 50, 'edge_style': 'bold', 'edge_width': 50, 'merge_regions_level': 'aggressive', 'simplification_level': 'high', 'contrast_boost': 50, 'detail_preservation': 'high', 'noise_reduction_level': 50, 'preprocessing_mode': 'artistic', 'sharpen_level': 50, 'eye_detection_sensitivity': 'high', 'face_detection_mode': 'ML', 'feature_detection_enabled': False, 'feature_importance': 50, 'feature_protection_radius': 50, 'preserve_expressions': False, 'color_grouping_threshold': 50, 'color_harmony': 'analogous', 'color_saturation_boost': 50, 'dark_area_enhancement': 50, 'highlight_preservation': 'high', 'light_area_protection': 50, 'min_number_size': 50, 'number_contrast': 'high', 'number_legibility_priority': 50, 'number_overlap_strategy': 'move', 'number_placement': 'avoid_features', 'number_size_strategy': 'adaptive'}
2025-03-22 22:48:20,526 - pbn-app.processor-pipeline - INFO - Starting processing pipeline for 1742676467_FELV-cat.jpg
2025-03-22 22:48:20,586 - pbn-app.processor-pipeline - INFO - Auto-detected image type: landscape
2025-03-22 22:48:20,587 - pbn-app.image_preprocessor - INFO - Starting image preprocessing
2025-03-22 22:48:20,700 - pbn-app.image_preprocessor - INFO - Image preprocessing completed
2025-03-22 22:48:20,700 - pbn-app.processor-pipeline - INFO - Preprocessing completed in 0.17s
2025-03-22 22:48:20,700 - pbn-app.processor-pipeline - ERROR - Error during feature detection: 'FeatureDetector' object has no attribute 'detect_features'
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\enhanced\processor_pipeline.py", line 96, in process_image
    feature_results = self.feature_detector.detect_features(processed_image, settings)
AttributeError: 'FeatureDetector' object has no attribute 'detect_features'
2025-03-22 22:48:20,701 - pbn-app.template_generator - INFO - Starting template generation
2025-03-22 22:48:20,701 - pbn-app.template_generator - INFO - Segmenting image with 30 colors, simplification 0.75
2025-03-22 22:48:23,540 - pbn-app.template_generator - INFO - Template generation completed in 2.84s
2025-03-22 22:48:23,540 - pbn-app.region_optimizer - INFO - Starting region optimization
2025-03-22 22:48:23,582 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:23] "POST /enhanced/api/analyze HTTP/1.1" 200 -
2025-03-22 22:48:23,618 - pbn-app.processor-pipeline - ERROR - Error during template generation: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\segmentation.cpp:161: error: (-215:Assertion failed) src.type() == CV_8UC3 && dst.type() == CV_32SC1 in function 'cv::watershed'
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\enhanced\processor_pipeline.py", line 117, in process_image
    optimized_segments = self.region_optimizer.optimize_regions(
  File "C:\PROJECTS\Paint by numbers\enhanced\region_optimizer.py", line 49, in optimize_regions
    segments = self._simplify_boundaries(segments, simplify_strength)
  File "C:\PROJECTS\Paint by numbers\enhanced\region_optimizer.py", line 209, in _simplify_boundaries
    cv2.watershed(np.zeros_like(segments)[:, :, np.newaxis], markers)
cv2.error: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\segmentation.cpp:161: error: (-215:Assertion failed) src.type() == CV_8UC3 && dst.type() == CV_32SC1 in function 'cv::watershed'

2025-03-22 22:48:23,632 - pbn-app.processor-pipeline - INFO - Final template creation completed in 3.11s
2025-03-22 22:48:23,632 - pbn-app.processor-pipeline - INFO - Total processing time: 3.11s
2025-03-22 22:48:23,633 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:23] "POST /enhanced/api/preview HTTP/1.1" 200 -
2025-03-22 22:48:23,641 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:23] "GET /static/previews/preview_1742676467_template.jpg?t=1742676503638 HTTP/1.1" 200 -
2025-03-22 22:48:29,040 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:29] "GET /uploads/preview_1742676467_template.jpg?t=1742676503638 HTTP/1.1" 500 -
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 917, in full_dispatch_request
    rv = self.dispatch_request()
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 891, in dispatch_request
    self.raise_routing_exception(req)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 500, in raise_routing_exception
    raise request.routing_exception  # type: ignore[misc]
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\ctx.py", line 362, in match_request
    result = self.url_adapter.match(return_rule=True)  # type: ignore
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\werkzeug\routing\map.py", line 629, in match
    raise NotFound() from None
werkzeug.exceptions.NotFound: 404 Not Found: The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1536, in __call__
    return self.wsgi_app(environ, start_response)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1514, in wsgi_app
    response = self.handle_exception(e)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1511, in wsgi_app
    response = self.full_dispatch_request()
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 919, in full_dispatch_request
    rv = self.handle_user_exception(e)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 802, in handle_user_exception
    return self.handle_http_exception(e)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 777, in handle_http_exception
    return self.ensure_sync(handler)(e)  # type: ignore[no-any-return]
  File "C:\PROJECTS\Paint by numbers\app.py", line 35, in page_not_found
    return render_template('error.html', error_code=404, error_message="Page not found"), 404
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\templating.py", line 149, in render_template
    template = app.jinja_env.get_or_select_template(template_name_or_list)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\environment.py", line 1087, in get_or_select_template
    return self.get_template(template_name_or_list, parent, globals)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\environment.py", line 1016, in get_template
    return self._load_template(name, globals)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\environment.py", line 975, in _load_template
    template = self.loader.load(self, name, self.make_globals(globals))
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\loaders.py", line 126, in load
    source, filename, uptodate = self.get_source(environment, name)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\templating.py", line 65, in get_source
    return self._get_source_fast(environment, template)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\templating.py", line 99, in _get_source_fast
    raise TemplateNotFound(template)
jinja2.exceptions.TemplateNotFound: error.html
2025-03-22 22:48:31,308 - pbn-app - INFO - Auto-detected image type: still_life (confidence: 0.53)
2025-03-22 22:48:31,308 - pbn-app.settings_manager - INFO - Applied custom parameters: {'colors': 50, 'edge_strength': 50, 'edge_style': 'bold', 'edge_width': 50, 'merge_regions_level': 'aggressive', 'simplification_level': 'high', 'contrast_boost': 50, 'detail_preservation': 'high', 'noise_reduction_level': 50, 'preprocessing_mode': 'artistic', 'sharpen_level': 50, 'eye_detection_sensitivity': 'high', 'face_detection_mode': 'ML', 'feature_detection_enabled': False, 'feature_importance': 50, 'feature_protection_radius': 50, 'preserve_expressions': False, 'color_grouping_threshold': 50, 'color_harmony': 'analogous', 'color_saturation_boost': 50, 'dark_area_enhancement': 50, 'highlight_preservation': 'high', 'light_area_protection': 50, 'min_number_size': 50, 'number_contrast': 'high', 'number_legibility_priority': 50, 'number_overlap_strategy': 'move', 'number_placement': 'avoid_features', 'number_size_strategy': 'adaptive'}
2025-03-22 22:48:31,308 - pbn-app.image_preprocessor - INFO - Starting image preprocessing
2025-03-22 22:48:31,322 - pbn-app.image_preprocessor - INFO - Image preprocessing completed
2025-03-22 22:48:31,332 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:31] "POST /enhanced/api/preview HTTP/1.1" 200 -
2025-03-22 22:48:31,335 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:31] "GET /static/previews/preview_1742676467_preprocessed.jpg?t=1742676511334 HTTP/1.1" 200 -
2025-03-22 22:48:33,072 - pbn-app - INFO - Auto-detected image type: still_life (confidence: 0.53)
2025-03-22 22:48:33,072 - pbn-app.settings_manager - INFO - Applied custom parameters: {'colors': 50, 'edge_strength': 50, 'edge_style': 'bold', 'edge_width': 50, 'merge_regions_level': 'aggressive', 'simplification_level': 'high', 'contrast_boost': 50, 'detail_preservation': 'high', 'noise_reduction_level': 50, 'preprocessing_mode': 'artistic', 'sharpen_level': 50, 'eye_detection_sensitivity': 'high', 'face_detection_mode': 'ML', 'feature_detection_enabled': False, 'feature_importance': 50, 'feature_protection_radius': 50, 'preserve_expressions': False, 'color_grouping_threshold': 50, 'color_harmony': 'analogous', 'color_saturation_boost': 50, 'dark_area_enhancement': 50, 'highlight_preservation': 'high', 'light_area_protection': 50, 'min_number_size': 50, 'number_contrast': 'high', 'number_legibility_priority': 50, 'number_overlap_strategy': 'move', 'number_placement': 'avoid_features', 'number_size_strategy': 'adaptive'}
2025-03-22 22:48:33,072 - pbn-app.processor-pipeline - INFO - Starting processing pipeline for 1742676467_FELV-cat.jpg
2025-03-22 22:48:33,124 - pbn-app.processor-pipeline - INFO - Auto-detected image type: landscape
2025-03-22 22:48:33,124 - pbn-app.image_preprocessor - INFO - Starting image preprocessing
2025-03-22 22:48:33,151 - pbn-app.image_preprocessor - INFO - Image preprocessing completed
2025-03-22 22:48:33,151 - pbn-app.processor-pipeline - INFO - Preprocessing completed in 0.08s
2025-03-22 22:48:33,151 - pbn-app.processor-pipeline - ERROR - Error during feature detection: 'FeatureDetector' object has no attribute 'detect_features'
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\enhanced\processor_pipeline.py", line 96, in process_image
    feature_results = self.feature_detector.detect_features(processed_image, settings)
AttributeError: 'FeatureDetector' object has no attribute 'detect_features'
2025-03-22 22:48:33,151 - pbn-app.template_generator - INFO - Starting template generation
2025-03-22 22:48:33,151 - pbn-app.template_generator - INFO - Segmenting image with 30 colors, simplification 0.75
2025-03-22 22:48:33,857 - pbn-app - INFO - Auto-detected image type: still_life (confidence: 0.53)
2025-03-22 22:48:33,857 - pbn-app.settings_manager - INFO - Applied custom parameters: {'colors': 50, 'edge_strength': 50, 'edge_style': 'bold', 'edge_width': 50, 'merge_regions_level': 'aggressive', 'simplification_level': 'high', 'contrast_boost': 50, 'detail_preservation': 'high', 'noise_reduction_level': 50, 'preprocessing_mode': 'artistic', 'sharpen_level': 50, 'eye_detection_sensitivity': 'high', 'face_detection_mode': 'ML', 'feature_detection_enabled': False, 'feature_importance': 50, 'feature_protection_radius': 50, 'preserve_expressions': False, 'color_grouping_threshold': 50, 'color_harmony': 'analogous', 'color_saturation_boost': 50, 'dark_area_enhancement': 50, 'highlight_preservation': 'high', 'light_area_protection': 50, 'min_number_size': 50, 'number_contrast': 'high', 'number_legibility_priority': 50, 'number_overlap_strategy': 'move', 'number_placement': 'avoid_features', 'number_size_strategy': 'adaptive'}
2025-03-22 22:48:33,857 - pbn-app.processor-pipeline - INFO - Starting processing pipeline for 1742676467_FELV-cat.jpg
2025-03-22 22:48:33,917 - pbn-app.processor-pipeline - INFO - Auto-detected image type: landscape
2025-03-22 22:48:33,917 - pbn-app.image_preprocessor - INFO - Starting image preprocessing
2025-03-22 22:48:33,925 - pbn-app.image_preprocessor - INFO - Image preprocessing completed
2025-03-22 22:48:33,925 - pbn-app.processor-pipeline - INFO - Preprocessing completed in 0.07s
2025-03-22 22:48:33,925 - pbn-app.processor-pipeline - ERROR - Error during feature detection: 'FeatureDetector' object has no attribute 'detect_features'
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\enhanced\processor_pipeline.py", line 96, in process_image
    feature_results = self.feature_detector.detect_features(processed_image, settings)
AttributeError: 'FeatureDetector' object has no attribute 'detect_features'
2025-03-22 22:48:33,925 - pbn-app.template_generator - INFO - Starting template generation
2025-03-22 22:48:33,925 - pbn-app.template_generator - INFO - Segmenting image with 30 colors, simplification 0.75
2025-03-22 22:48:35,589 - pbn-app - INFO - Auto-detected image type: still_life (confidence: 0.53)
2025-03-22 22:48:35,589 - pbn-app.settings_manager - INFO - Applied custom parameters: {'colors': 50, 'edge_strength': 50, 'edge_style': 'bold', 'edge_width': 50, 'merge_regions_level': 'aggressive', 'simplification_level': 'high', 'contrast_boost': 50, 'detail_preservation': 'high', 'noise_reduction_level': 50, 'preprocessing_mode': 'artistic', 'sharpen_level': 50, 'eye_detection_sensitivity': 'high', 'face_detection_mode': 'ML', 'feature_detection_enabled': False, 'feature_importance': 50, 'feature_protection_radius': 50, 'preserve_expressions': False, 'color_grouping_threshold': 50, 'color_harmony': 'analogous', 'color_saturation_boost': 50, 'dark_area_enhancement': 50, 'highlight_preservation': 'high', 'light_area_protection': 50, 'min_number_size': 50, 'number_contrast': 'high', 'number_legibility_priority': 50, 'number_overlap_strategy': 'move', 'number_placement': 'avoid_features', 'number_size_strategy': 'adaptive'}
2025-03-22 22:48:35,589 - pbn-app.processor-pipeline - INFO - Starting processing pipeline for 1742676467_FELV-cat.jpg
2025-03-22 22:48:35,838 - pbn-app.processor-pipeline - INFO - Auto-detected image type: landscape
2025-03-22 22:48:35,838 - pbn-app.image_preprocessor - INFO - Starting image preprocessing
2025-03-22 22:48:35,872 - pbn-app.image_preprocessor - INFO - Image preprocessing completed
2025-03-22 22:48:35,873 - pbn-app.processor-pipeline - INFO - Preprocessing completed in 0.28s
2025-03-22 22:48:35,873 - pbn-app.processor-pipeline - ERROR - Error during feature detection: 'FeatureDetector' object has no attribute 'detect_features'
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\enhanced\processor_pipeline.py", line 96, in process_image
    feature_results = self.feature_detector.detect_features(processed_image, settings)
AttributeError: 'FeatureDetector' object has no attribute 'detect_features'
2025-03-22 22:48:35,873 - pbn-app.template_generator - INFO - Starting template generation
2025-03-22 22:48:35,873 - pbn-app.template_generator - INFO - Segmenting image with 30 colors, simplification 0.75
2025-03-22 22:48:36,142 - pbn-app - INFO - Auto-detected image type: still_life (confidence: 0.53)
2025-03-22 22:48:36,145 - pbn-app.settings_manager - INFO - Applied custom parameters: {'colors': 50, 'edge_strength': 50, 'edge_style': 'bold', 'edge_width': 50, 'merge_regions_level': 'aggressive', 'simplification_level': 'high', 'contrast_boost': 50, 'detail_preservation': 'high', 'noise_reduction_level': 50, 'preprocessing_mode': 'artistic', 'sharpen_level': 50, 'eye_detection_sensitivity': 'high', 'face_detection_mode': 'ML', 'feature_detection_enabled': False, 'feature_importance': 50, 'feature_protection_radius': 50, 'preserve_expressions': False, 'color_grouping_threshold': 50, 'color_harmony': 'analogous', 'color_saturation_boost': 50, 'dark_area_enhancement': 50, 'highlight_preservation': 'high', 'light_area_protection': 50, 'min_number_size': 50, 'number_contrast': 'high', 'number_legibility_priority': 50, 'number_overlap_strategy': 'move', 'number_placement': 'avoid_features', 'number_size_strategy': 'adaptive'}
2025-03-22 22:48:36,147 - pbn-app.image_preprocessor - INFO - Starting image preprocessing
2025-03-22 22:48:36,169 - pbn-app.image_preprocessor - INFO - Image preprocessing completed
2025-03-22 22:48:36,171 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:36] "POST /enhanced/api/preview HTTP/1.1" 200 -
2025-03-22 22:48:36,179 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:36] "GET /static/previews/preview_1742676467_preprocessed.jpg?t=1742676516174 HTTP/1.1" 200 -      
2025-03-22 22:48:36,857 - pbn-app - INFO - Auto-detected image type: still_life (confidence: 0.53)
2025-03-22 22:48:36,857 - pbn-app.settings_manager - INFO - Applied custom parameters: {'colors': 50, 'edge_strength': 50, 'edge_style': 'bold', 'edge_width': 50, 'merge_regions_level': 'aggressive', 'simplification_level': 'high', 'contrast_boost': 50, 'detail_preservation': 'high', 'noise_reduction_level': 50, 'preprocessing_mode': 'artistic', 'sharpen_level': 50, 'eye_detection_sensitivity': 'high', 'face_detection_mode': 'ML', 'feature_detection_enabled': False, 'feature_importance': 50, 'feature_protection_radius': 50, 'preserve_expressions': False, 'color_grouping_threshold': 50, 'color_harmony': 'analogous', 'color_saturation_boost': 50, 'dark_area_enhancement': 50, 'highlight_preservation': 'high', 'light_area_protection': 50, 'min_number_size': 50, 'number_contrast': 'high', 'number_legibility_priority': 50, 'number_overlap_strategy': 'move', 'number_placement': 'avoid_features', 'number_size_strategy': 'adaptive'}
2025-03-22 22:48:36,857 - pbn-app.image_preprocessor - INFO - Starting image preprocessing
2025-03-22 22:48:36,881 - pbn-app.image_preprocessor - INFO - Image preprocessing completed
2025-03-22 22:48:36,883 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:36] "POST /enhanced/api/preview HTTP/1.1" 200 -
2025-03-22 22:48:36,892 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:36] "GET /static/previews/preview_1742676467_preprocessed.jpg?t=1742676516886 HTTP/1.1" 200 -
2025-03-22 22:48:37,627 - pbn-app.template_generator - INFO - Template generation completed in 4.48s
2025-03-22 22:48:37,627 - pbn-app.region_optimizer - INFO - Starting region optimization
2025-03-22 22:48:37,762 - pbn-app.processor-pipeline - ERROR - Error during template generation: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\segmentation.cpp:161: error: (-215:Assertion failed) src.type() == CV_8UC3 && dst.type() == CV_32SC1 in function 'cv::watershed'
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\enhanced\processor_pipeline.py", line 117, in process_image
    optimized_segments = self.region_optimizer.optimize_regions(
  File "C:\PROJECTS\Paint by numbers\enhanced\region_optimizer.py", line 49, in optimize_regions
    segments = self._simplify_boundaries(segments, simplify_strength)
  File "C:\PROJECTS\Paint by numbers\enhanced\region_optimizer.py", line 209, in _simplify_boundaries
    cv2.watershed(np.zeros_like(segments)[:, :, np.newaxis], markers)
cv2.error: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\segmentation.cpp:161: error: (-215:Assertion failed) src.type() == CV_8UC3 && dst.type() == CV_32SC1 in function 'cv::watershed'

2025-03-22 22:48:37,789 - pbn-app.processor-pipeline - INFO - Final template creation completed in 4.72s
2025-03-22 22:48:37,789 - pbn-app.processor-pipeline - INFO - Total processing time: 4.72s
2025-03-22 22:48:37,795 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:37] "POST /enhanced/api/preview HTTP/1.1" 200 -
2025-03-22 22:48:37,802 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:37] "GET /static/previews/preview_1742676467_segments.jpg?t=1742676517796 HTTP/1.1" 200 -
2025-03-22 22:48:39,522 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:39] "GET /uploads/preview_1742676467_segments.jpg?t=1742676517796 HTTP/1.1" 500 -
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 917, in full_dispatch_request
    rv = self.dispatch_request()
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 891, in dispatch_request
    self.raise_routing_exception(req)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 500, in raise_routing_exception
    raise request.routing_exception  # type: ignore[misc]
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\ctx.py", line 362, in match_request
    result = self.url_adapter.match(return_rule=True)  # type: ignore
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\werkzeug\routing\map.py", line 629, in match
    raise NotFound() from None
werkzeug.exceptions.NotFound: 404 Not Found: The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1536, in __call__
    return self.wsgi_app(environ, start_response)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1514, in wsgi_app
    response = self.handle_exception(e)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1511, in wsgi_app
    response = self.full_dispatch_request()
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 919, in full_dispatch_request
    rv = self.handle_user_exception(e)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 802, in handle_user_exception
    return self.handle_http_exception(e)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 777, in handle_http_exception
    return self.ensure_sync(handler)(e)  # type: ignore[no-any-return]
  File "C:\PROJECTS\Paint by numbers\app.py", line 35, in page_not_found
    return render_template('error.html', error_code=404, error_message="Page not found"), 404
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\templating.py", line 149, in render_template
    template = app.jinja_env.get_or_select_template(template_name_or_list)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\environment.py", line 1087, in get_or_select_template
    return self.get_template(template_name_or_list, parent, globals)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\environment.py", line 1016, in get_template
    return self._load_template(name, globals)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\environment.py", line 975, in _load_template
    template = self.loader.load(self, name, self.make_globals(globals))
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\loaders.py", line 126, in load
    source, filename, uptodate = self.get_source(environment, name)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\templating.py", line 65, in get_source
    return self._get_source_fast(environment, template)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\templating.py", line 99, in _get_source_fast
    raise TemplateNotFound(template)
jinja2.exceptions.TemplateNotFound: error.html
2025-03-22 22:48:39,669 - pbn-app.template_generator - INFO - Template generation completed in 5.74s
2025-03-22 22:48:39,669 - pbn-app.region_optimizer - INFO - Starting region optimization
2025-03-22 22:48:39,782 - pbn-app.processor-pipeline - ERROR - Error during template generation: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\segmentation.cpp:161: error: (-215:Assertion failed) src.type() == CV_8UC3 && dst.type() == CV_32SC1 in function 'cv::watershed'
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\enhanced\processor_pipeline.py", line 117, in process_image
    optimized_segments = self.region_optimizer.optimize_regions(
  File "C:\PROJECTS\Paint by numbers\enhanced\region_optimizer.py", line 49, in optimize_regions
    segments = self._simplify_boundaries(segments, simplify_strength)
  File "C:\PROJECTS\Paint by numbers\enhanced\region_optimizer.py", line 209, in _simplify_boundaries
    cv2.watershed(np.zeros_like(segments)[:, :, np.newaxis], markers)
cv2.error: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\segmentation.cpp:161: error: (-215:Assertion failed) src.type() == CV_8UC3 && dst.type() == CV_32SC1 in function 'cv::watershed'

2025-03-22 22:48:39,804 - pbn-app.processor-pipeline - INFO - Final template creation completed in 5.95s
2025-03-22 22:48:39,804 - pbn-app.processor-pipeline - INFO - Total processing time: 5.95s
2025-03-22 22:48:39,821 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:39] "POST /enhanced/api/preview HTTP/1.1" 500 -
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1536, in __call__
    return self.wsgi_app(environ, start_response)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1514, in wsgi_app
    response = self.handle_exception(e)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1511, in wsgi_app
    response = self.full_dispatch_request()
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 919, in full_dispatch_request
    rv = self.handle_user_exception(e)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 917, in full_dispatch_request
    rv = self.dispatch_request()
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 902, in dispatch_request
    return self.ensure_sync(self.view_functions[rule.endpoint])(**view_args)  # type: ignore[no-any-return]
  File "C:\PROJECTS\Paint by numbers\enhanced_ui.py", line 123, in generate_preview
    preview_image = cv2.cvtColor(preview_result['preview'], cv2.COLOR_RGB2BGR)
cv2.error: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\color.cpp:199: error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'    

2025-03-22 22:48:40,726 - pbn-app.template_generator - INFO - Template generation completed in 4.85s
2025-03-22 22:48:40,727 - pbn-app.region_optimizer - INFO - Starting region optimization
2025-03-22 22:48:40,795 - pbn-app.processor-pipeline - ERROR - Error during template generation: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\segmentation.cpp:161: error: (-215:Assertion failed) src.type() == CV_8UC3 && dst.type() == CV_32SC1 in function 'cv::watershed'
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\enhanced\processor_pipeline.py", line 117, in process_image
    optimized_segments = self.region_optimizer.optimize_regions(
  File "C:\PROJECTS\Paint by numbers\enhanced\region_optimizer.py", line 49, in optimize_regions
    segments = self._simplify_boundaries(segments, simplify_strength)
  File "C:\PROJECTS\Paint by numbers\enhanced\region_optimizer.py", line 209, in _simplify_boundaries
    cv2.watershed(np.zeros_like(segments)[:, :, np.newaxis], markers)
cv2.error: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\segmentation.cpp:161: error: (-215:Assertion failed) src.type() == CV_8UC3 && dst.type() == CV_32SC1 in function 'cv::watershed'

2025-03-22 22:48:40,805 - pbn-app.processor-pipeline - INFO - Final template creation completed in 5.22s
2025-03-22 22:48:40,805 - pbn-app.processor-pipeline - INFO - Total processing time: 5.22s
2025-03-22 22:48:40,805 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:40] "POST /enhanced/api/preview HTTP/1.1" 200 -
2025-03-22 22:48:40,814 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:40] "GET /static/previews/preview_1742676467_segments.jpg?t=1742676520811 HTTP/1.1" 200 -
2025-03-22 22:48:41,045 - pbn-app - INFO - Auto-detected image type: still_life (confidence: 0.53)
2025-03-22 22:48:41,045 - pbn-app.settings_manager - INFO - Applied custom parameters: {'colors': 50, 'edge_strength': 50, 'edge_style': 'bold', 'edge_width': 50, 'merge_regions_level': 'aggressive', 'simplification_level': 'high', 'contrast_boost': 50, 'detail_preservation': 'high', 'noise_reduction_level': 50, 'preprocessing_mode': 'artistic', 'sharpen_level': 50, 'eye_detection_sensitivity': 'high', 'face_detection_mode': 'ML', 'feature_detection_enabled': False, 'feature_importance': 50, 'feature_protection_radius': 50, 'preserve_expressions': False, 'color_grouping_threshold': 50, 'color_harmony': 'analogous', 'color_saturation_boost': 50, 'dark_area_enhancement': 50, 'highlight_preservation': 'high', 'light_area_protection': 50, 'min_number_size': 50, 'number_contrast': 'high', 'number_legibility_priority': 50, 'number_overlap_strategy': 'move', 'number_placement': 'avoid_features', 'number_size_strategy': 'adaptive'}
2025-03-22 22:48:41,045 - pbn-app.processor-pipeline - INFO - Starting processing pipeline for 1742676467_FELV-cat.jpg
2025-03-22 22:48:41,094 - pbn-app.processor-pipeline - INFO - Auto-detected image type: landscape
2025-03-22 22:48:41,094 - pbn-app.image_preprocessor - INFO - Starting image preprocessing
2025-03-22 22:48:41,105 - pbn-app.image_preprocessor - INFO - Image preprocessing completed
2025-03-22 22:48:41,105 - pbn-app.processor-pipeline - INFO - Preprocessing completed in 0.06s
2025-03-22 22:48:41,105 - pbn-app.processor-pipeline - ERROR - Error during feature detection: 'FeatureDetector' object has no attribute 'detect_features'
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\enhanced\processor_pipeline.py", line 96, in process_image
    feature_results = self.feature_detector.detect_features(processed_image, settings)
AttributeError: 'FeatureDetector' object has no attribute 'detect_features'
2025-03-22 22:48:41,105 - pbn-app.template_generator - INFO - Starting template generation
2025-03-22 22:48:41,105 - pbn-app.template_generator - INFO - Segmenting image with 30 colors, simplification 0.75
2025-03-22 22:48:41,982 - pbn-app - INFO - Auto-detected image type: still_life (confidence: 0.53)
2025-03-22 22:48:41,984 - pbn-app.settings_manager - INFO - Applied custom parameters: {'colors': 50, 'edge_strength': 50, 'edge_style': 'bold', 'edge_width': 50, 'merge_regions_level': 'aggressive', 'simplification_level': 'high', 'contrast_boost': 50, 'detail_preservation': 'high', 'noise_reduction_level': 50, 'preprocessing_mode': 'artistic', 'sharpen_level': 50, 'eye_detection_sensitivity': 'high', 'face_detection_mode': 'ML', 'feature_detection_enabled': False, 'feature_importance': 50, 'feature_protection_radius': 50, 'preserve_expressions': False, 'color_grouping_threshold': 50, 'color_harmony': 'analogous', 'color_saturation_boost': 50, 'dark_area_enhancement': 50, 'highlight_preservation': 'high', 'light_area_protection': 50, 'min_number_size': 50, 'number_contrast': 'high', 'number_legibility_priority': 50, 'number_overlap_strategy': 'move', 'number_placement': 'avoid_features', 'number_size_strategy': 'adaptive'}
2025-03-22 22:48:41,984 - pbn-app.image_preprocessor - INFO - Starting image preprocessing
2025-03-22 22:48:42,005 - pbn-app.image_preprocessor - INFO - Image preprocessing completed
2025-03-22 22:48:42,008 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:42] "POST /enhanced/api/preview HTTP/1.1" 200 -
2025-03-22 22:48:42,012 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:42] "GET /static/previews/preview_1742676467_preprocessed.jpg?t=1742676522009 HTTP/1.1" 200 -      
2025-03-22 22:48:42,566 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:42] "GET /uploads/preview_1742676467_preprocessed.jpg?t=1742676522009 HTTP/1.1" 500 -
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 917, in full_dispatch_request
    rv = self.dispatch_request()
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 891, in dispatch_request
    self.raise_routing_exception(req)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 500, in raise_routing_exception
    raise request.routing_exception  # type: ignore[misc]
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\ctx.py", line 362, in match_request
    result = self.url_adapter.match(return_rule=True)  # type: ignore
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\werkzeug\routing\map.py", line 629, in match
    raise NotFound() from None
werkzeug.exceptions.NotFound: 404 Not Found: The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1536, in __call__
    return self.wsgi_app(environ, start_response)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1514, in wsgi_app
    response = self.handle_exception(e)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 1511, in wsgi_app
    response = self.full_dispatch_request()
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 919, in full_dispatch_request
    rv = self.handle_user_exception(e)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 802, in handle_user_exception
    return self.handle_http_exception(e)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\app.py", line 777, in handle_http_exception
    return self.ensure_sync(handler)(e)  # type: ignore[no-any-return]
  File "C:\PROJECTS\Paint by numbers\app.py", line 35, in page_not_found
    return render_template('error.html', error_code=404, error_message="Page not found"), 404
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\templating.py", line 149, in render_template
    template = app.jinja_env.get_or_select_template(template_name_or_list)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\environment.py", line 1087, in get_or_select_template
    return self.get_template(template_name_or_list, parent, globals)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\environment.py", line 1016, in get_template
    return self._load_template(name, globals)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\environment.py", line 975, in _load_template
    template = self.loader.load(self, name, self.make_globals(globals))
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\jinja2\loaders.py", line 126, in load
    source, filename, uptodate = self.get_source(environment, name)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\templating.py", line 65, in get_source
    return self._get_source_fast(environment, template)
  File "C:\PROJECTS\Paint by numbers\venv\lib\site-packages\flask\templating.py", line 99, in _get_source_fast
    raise TemplateNotFound(template)
jinja2.exceptions.TemplateNotFound: error.html
2025-03-22 22:48:43,227 - pbn-app - INFO - Auto-detected image type: still_life (confidence: 0.53)
2025-03-22 22:48:43,229 - pbn-app.settings_manager - INFO - Applied custom parameters: {'colors': 50, 'edge_strength': 50, 'edge_style': 'bold', 'edge_width': 50, 'merge_regions_level': 'aggressive', 'simplification_level': 'high', 'contrast_boost': 50, 'detail_preservation': 'high', 'noise_reduction_level': 50, 'preprocessing_mode': 'artistic', 'sharpen_level': 50, 'eye_detection_sensitivity': 'high', 'face_detection_mode': 'ML', 'feature_detection_enabled': False, 'feature_importance': 50, 'feature_protection_radius': 50, 'preserve_expressions': False, 'color_grouping_threshold': 50, 'color_harmony': 'analogous', 'color_saturation_boost': 50, 'dark_area_enhancement': 50, 'highlight_preservation': 'high', 'light_area_protection': 50, 'min_number_size': 50, 'number_contrast': 'high', 'number_legibility_priority': 50, 'number_overlap_strategy': 'move', 'number_placement': 'avoid_features', 'number_size_strategy': 'adaptive'}
2025-03-22 22:48:43,230 - pbn-app.processor-pipeline - INFO - Starting processing pipeline for 1742676467_FELV-cat.jpg
2025-03-22 22:48:43,439 - pbn-app.processor-pipeline - INFO - Auto-detected image type: landscape
2025-03-22 22:48:43,439 - pbn-app.image_preprocessor - INFO - Starting image preprocessing
2025-03-22 22:48:43,462 - pbn-app.image_preprocessor - INFO - Image preprocessing completed
2025-03-22 22:48:43,462 - pbn-app.processor-pipeline - INFO - Preprocessing completed in 0.23s
2025-03-22 22:48:43,462 - pbn-app.processor-pipeline - ERROR - Error during feature detection: 'FeatureDetector' object has no attribute 'detect_features'
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\enhanced\processor_pipeline.py", line 96, in process_image
    feature_results = self.feature_detector.detect_features(processed_image, settings)
AttributeError: 'FeatureDetector' object has no attribute 'detect_features'
2025-03-22 22:48:43,462 - pbn-app.template_generator - INFO - Starting template generation
2025-03-22 22:48:43,462 - pbn-app.template_generator - INFO - Segmenting image with 30 colors, simplification 0.75
2025-03-22 22:48:44,200 - pbn-app.template_generator - INFO - Template generation completed in 3.09s
2025-03-22 22:48:44,201 - pbn-app.region_optimizer - INFO - Starting region optimization
2025-03-22 22:48:44,284 - pbn-app.processor-pipeline - ERROR - Error during template generation: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\segmentation.cpp:161: error: (-215:Assertion failed) src.type() == CV_8UC3 && dst.type() == CV_32SC1 in function 'cv::watershed'
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\enhanced\processor_pipeline.py", line 117, in process_image
    optimized_segments = self.region_optimizer.optimize_regions(
  File "C:\PROJECTS\Paint by numbers\enhanced\region_optimizer.py", line 49, in optimize_regions
    segments = self._simplify_boundaries(segments, simplify_strength)
  File "C:\PROJECTS\Paint by numbers\enhanced\region_optimizer.py", line 209, in _simplify_boundaries
    cv2.watershed(np.zeros_like(segments)[:, :, np.newaxis], markers)
cv2.error: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\segmentation.cpp:161: error: (-215:Assertion failed) src.type() == CV_8UC3 && dst.type() == CV_32SC1 in function 'cv::watershed'

2025-03-22 22:48:44,297 - pbn-app.processor-pipeline - INFO - Final template creation completed in 3.25s
2025-03-22 22:48:44,297 - pbn-app.processor-pipeline - INFO - Total processing time: 3.25s
2025-03-22 22:48:44,301 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:44] "POST /enhanced/api/preview HTTP/1.1" 200 -
2025-03-22 22:48:44,303 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:44] "GET /static/previews/preview_1742676467_segments.jpg?t=1742676524303 HTTP/1.1" 200 -
2025-03-22 22:48:46,651 - pbn-app.template_generator - INFO - Template generation completed in 3.19s
2025-03-22 22:48:46,651 - pbn-app.region_optimizer - INFO - Starting region optimization
2025-03-22 22:48:46,713 - pbn-app.processor-pipeline - ERROR - Error during template generation: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\segmentation.cpp:161: error: (-215:Assertion failed) src.type() == CV_8UC3 && dst.type() == CV_32SC1 in function 'cv::watershed'
Traceback (most recent call last):
  File "C:\PROJECTS\Paint by numbers\enhanced\processor_pipeline.py", line 117, in process_image
    optimized_segments = self.region_optimizer.optimize_regions(
  File "C:\PROJECTS\Paint by numbers\enhanced\region_optimizer.py", line 49, in optimize_regions
    segments = self._simplify_boundaries(segments, simplify_strength)
  File "C:\PROJECTS\Paint by numbers\enhanced\region_optimizer.py", line 209, in _simplify_boundaries
    cv2.watershed(np.zeros_like(segments)[:, :, np.newaxis], markers)
cv2.error: OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\segmentation.cpp:161: error: (-215:Assertion failed) src.type() == CV_8UC3 && dst.type() == CV_32SC1 in function 'cv::watershed'

2025-03-22 22:48:46,729 - pbn-app.processor-pipeline - INFO - Final template creation completed in 3.50s
2025-03-22 22:48:46,730 - pbn-app.processor-pipeline - INFO - Total processing time: 3.50s
2025-03-22 22:48:46,732 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:46] "POST /enhanced/api/preview HTTP/1.1" 200 -
2025-03-22 22:48:46,736 - werkzeug - INFO - 127.0.0.1 - - [22/Mar/2025 22:48:46] "GET /static/previews/preview_1742676467_template.jpg?t=1742676526734 HTTP/1.1" 200 -
