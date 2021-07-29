(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("e6dca842-cfcf-494e-a114-10b591ed8136");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'e6dca842-cfcf-494e-a114-10b591ed8136' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error(url) {
          console.error("failed to load " + url);
        }
    
        for (let i = 0; i < css_urls.length; i++) {
          const url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error.bind(null, url);
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.3.1.min.js": "YF85VygJKMVnHE+lLv2AM93Vbstr0yo2TbIu5v8se5Rq3UQAUmcuh4aaJwNlpKwa", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.3.1.min.js": "KKuas3gevv3PvrlkyCMzffFeaMq5we/a2QsP5AUoS3mJ0jmaCL7jirFJN3GoE/lM", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.3.1.min.js": "MK/uFc3YT18pkvvXRl66tTHjP0/dxoSH2e/eiNMFIguKlun2+WVqaPTWmUy/zvh4"};
    
        for (let i = 0; i < js_urls.length; i++) {
          const url = js_urls[i];
          const element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error.bind(null, url);
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.3.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.3.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.3.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"75f761a9-1715-4b3d-a365-d76fd309014e":{"defs":[],"roots":{"references":[{"attributes":{"active_multi":null,"tools":[{"id":"1226"},{"id":"1227"},{"id":"1228"},{"id":"1229"},{"id":"1230"},{"id":"1231"}]},"id":"1233","type":"Toolbar"},{"attributes":{},"id":"1267","type":"AllLabels"},{"attributes":{},"id":"1264","type":"StringEditor"},{"attributes":{},"id":"1229","type":"SaveTool"},{"attributes":{},"id":"1230","type":"ResetTool"},{"attributes":{},"id":"1231","type":"HelpTool"},{"attributes":{"below":[{"id":"1218"}],"center":[{"id":"1221"},{"id":"1225"},{"id":"1251"},{"id":"1252"}],"left":[{"id":"1222"}],"match_aspect":true,"renderers":[{"id":"1243"}],"title":{"id":"1258"},"toolbar":{"id":"1233"},"x_range":{"id":"1210"},"x_scale":{"id":"1214"},"y_range":{"id":"1212"},"y_scale":{"id":"1216"}},"id":"1209","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"1216","type":"LinearScale"},{"attributes":{"data_source":{"id":"1240"},"glyph":{"id":"1241"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1242"},"view":{"id":"1244"}},"id":"1243","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"blue"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"1242","type":"Circle"},{"attributes":{},"id":"1273","type":"UnionRenderers"},{"attributes":{"source":{"id":"1245"}},"id":"1250","type":"CDSView"},{"attributes":{"source":{"id":"1240"}},"id":"1244","type":"CDSView"},{"attributes":{"children":[{"id":"1209"}]},"id":"1256","type":"Row"},{"attributes":{},"id":"1212","type":"DataRange1d"},{"attributes":{},"id":"1258","type":"Title"},{"attributes":{},"id":"1214","type":"LinearScale"},{"attributes":{"columns":[{"id":"1246"},{"id":"1247"}],"header_row":false,"height":100,"index_position":null,"source":{"id":"1245"},"view":{"id":"1250"},"width":300},"id":"1248","type":"DataTable"},{"attributes":{"args":{"source":{"id":"1245"},"sx":{"id":"1251"},"sy":{"id":"1252"}},"code":"\\n\\n    var angle = cb_obj.value*Math.PI/180;\\n    sx.gradient=Math.tan(angle);\\n    sy.gradient=Math.tan(angle + Math.PI/2);\\n    var c00 = 1;\\n    var c10 = 2;\\n    var c11 = 5;\\n\\n    var ca = Math.cos(-angle);\\n    var sa = Math.sin(-angle);\\n    var ca2 = ca*ca;\\n    var sa2 = sa*sa;\\n    var sca = sa*ca;\\n\\n    var c00r = ca2*c00 + sa2*c11 - 2*sca*c10;\\n    var c01r = sca*(c00 - c11) + (ca2 - sa2)*c10;\\n    var c10r = c01r;\\n    var c11r = sa2*c00 + ca2*c11 + 2*sca*c10;\\n\\n    source.data[&#x27;c1&#x27;] = [c00r, c10r];\\n    source.data[&#x27;c2&#x27;] = [c01r, c11r];\\n    source.change.emit();\\n"},"id":"1254","type":"CustomJS"},{"attributes":{},"id":"1272","type":"Selection"},{"attributes":{"children":[{"id":"1255"},{"id":"1256"}]},"id":"1257","type":"Column"},{"attributes":{},"id":"1269","type":"BasicTickFormatter"},{"attributes":{"gradient":0.0,"line_color":"orange","line_dash":[6],"line_width":3.5,"y_intercept":0},"id":"1251","type":"Slope"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"1232","type":"BoxAnnotation"},{"attributes":{"high":90.0,"js_property_callbacks":{"change:value":[{"id":"1254"}]},"low":0.0,"step":5,"title":"Rotation angle,    Cov Mat:","value":0,"value_throttled":0,"width":200},"id":"1253","type":"Spinner"},{"attributes":{"gradient":1.633123935319537e+16,"line_color":"orange","line_dash":[6],"line_width":3.5,"y_intercept":0},"id":"1252","type":"Slope"},{"attributes":{"editor":{"id":"1262"},"field":"c1","formatter":{"id":"1263"}},"id":"1246","type":"TableColumn"},{"attributes":{"data":{"c1":[1,2],"c2":[2,5]},"selected":{"id":"1260"},"selection_policy":{"id":"1261"}},"id":"1245","type":"ColumnDataSource"},{"attributes":{},"id":"1260","type":"Selection"},{"attributes":{},"id":"1261","type":"UnionRenderers"},{"attributes":{},"id":"1219","type":"BasicTicker"},{"attributes":{"formatter":{"id":"1266"},"major_label_policy":{"id":"1267"},"ticker":{"id":"1219"}},"id":"1218","type":"LinearAxis"},{"attributes":{"children":[{"id":"1253"},{"id":"1248"}]},"id":"1255","type":"Row"},{"attributes":{},"id":"1262","type":"StringEditor"},{"attributes":{"editor":{"id":"1264"},"field":"c2","formatter":{"id":"1265"}},"id":"1247","type":"TableColumn"},{"attributes":{},"id":"1263","type":"StringFormatter"},{"attributes":{"axis":{"id":"1218"},"ticker":null},"id":"1221","type":"Grid"},{"attributes":{},"id":"1270","type":"AllLabels"},{"attributes":{"overlay":{"id":"1232"}},"id":"1228","type":"BoxZoomTool"},{"attributes":{},"id":"1210","type":"DataRange1d"},{"attributes":{"fill_color":{"value":"blue"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"1241","type":"Circle"},{"attributes":{"formatter":{"id":"1269"},"major_label_policy":{"id":"1270"},"ticker":{"id":"1223"}},"id":"1222","type":"LinearAxis"},{"attributes":{},"id":"1265","type":"StringFormatter"},{"attributes":{},"id":"1226","type":"PanTool"},{"attributes":{},"id":"1227","type":"WheelZoomTool"},{"attributes":{"axis":{"id":"1222"},"dimension":1,"ticker":null},"id":"1225","type":"Grid"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[1000]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[1000]}},"selected":{"id":"1272"},"selection_policy":{"id":"1273"}},"id":"1240","type":"ColumnDataSource"},{"attributes":{},"id":"1266","type":"BasicTickFormatter"},{"attributes":{},"id":"1223","type":"BasicTicker"}],"root_ids":["1257"]},"title":"Bokeh Application","version":"2.3.1"}}';
                  var render_items = [{"docid":"75f761a9-1715-4b3d-a365-d76fd309014e","root_ids":["1257"],"roots":{"1257":"e6dca842-cfcf-494e-a114-10b591ed8136"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();