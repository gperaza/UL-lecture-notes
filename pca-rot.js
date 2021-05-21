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
    
      
      
    
      var element = document.getElementById("65f7bd09-d4f1-469e-b724-4de2189c2e59");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '65f7bd09-d4f1-469e-b724-4de2189c2e59' but no matching script tag was found.")
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
                    
                  var docs_json = '{"d4511a28-9eaf-4b1a-abe0-ebee0872d6cb":{"defs":[],"roots":{"references":[{"attributes":{},"id":"3168","type":"BasicTickFormatter"},{"attributes":{},"id":"3111","type":"DataRange1d"},{"attributes":{"formatter":{"id":"3165"},"major_label_policy":{"id":"3167"},"ticker":{"id":"3122"}},"id":"3121","type":"LinearAxis"},{"attributes":{},"id":"3118","type":"BasicTicker"},{"attributes":{"children":[{"id":"3154"},{"id":"3155"}]},"id":"3156","type":"Column"},{"attributes":{},"id":"3161","type":"StringFormatter"},{"attributes":{},"id":"3170","type":"AllLabels"},{"attributes":{},"id":"3173","type":"Selection"},{"attributes":{},"id":"3130","type":"HelpTool"},{"attributes":{},"id":"3163","type":"StringFormatter"},{"attributes":{"columns":[{"id":"3145"},{"id":"3146"}],"header_row":false,"height":100,"index_position":null,"source":{"id":"3144"},"view":{"id":"3149"},"width":300},"id":"3147","type":"DataTable"},{"attributes":{"editor":{"id":"3164"},"field":"c2","formatter":{"id":"3163"}},"id":"3146","type":"TableColumn"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"blue"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"3141","type":"Circle"},{"attributes":{"children":[{"id":"3108"}]},"id":"3155","type":"Row"},{"attributes":{"editor":{"id":"3162"},"field":"c1","formatter":{"id":"3161"}},"id":"3145","type":"TableColumn"},{"attributes":{},"id":"3109","type":"DataRange1d"},{"attributes":{},"id":"3129","type":"ResetTool"},{"attributes":{},"id":"3115","type":"LinearScale"},{"attributes":{},"id":"3172","type":"UnionRenderers"},{"attributes":{},"id":"3113","type":"LinearScale"},{"attributes":{"active_multi":null,"tools":[{"id":"3125"},{"id":"3126"},{"id":"3127"},{"id":"3128"},{"id":"3129"},{"id":"3130"}]},"id":"3132","type":"Toolbar"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[1000]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[1000]}},"selected":{"id":"3173"},"selection_policy":{"id":"3172"}},"id":"3139","type":"ColumnDataSource"},{"attributes":{},"id":"3162","type":"StringEditor"},{"attributes":{},"id":"3157","type":"Title"},{"attributes":{"fill_color":{"value":"blue"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"3140","type":"Circle"},{"attributes":{"source":{"id":"3144"}},"id":"3149","type":"CDSView"},{"attributes":{"high":90.0,"js_property_callbacks":{"change:value":[{"id":"3153"}]},"low":0.0,"step":5,"title":"Rotation angle,    Cov Mat:","value":0,"value_throttled":0,"width":200},"id":"3152","type":"Spinner"},{"attributes":{},"id":"3125","type":"PanTool"},{"attributes":{},"id":"3126","type":"WheelZoomTool"},{"attributes":{"children":[{"id":"3152"},{"id":"3147"}]},"id":"3154","type":"Row"},{"attributes":{"gradient":1.633123935319537e+16,"line_color":"orange","line_dash":[6],"line_width":3.5,"y_intercept":0},"id":"3151","type":"Slope"},{"attributes":{},"id":"3159","type":"UnionRenderers"},{"attributes":{},"id":"3122","type":"BasicTicker"},{"attributes":{},"id":"3165","type":"BasicTickFormatter"},{"attributes":{"data":{"c1":[1,2],"c2":[2,5]},"selected":{"id":"3160"},"selection_policy":{"id":"3159"}},"id":"3144","type":"ColumnDataSource"},{"attributes":{},"id":"3164","type":"StringEditor"},{"attributes":{},"id":"3128","type":"SaveTool"},{"attributes":{"args":{"source":{"id":"3144"},"sx":{"id":"3150"},"sy":{"id":"3151"}},"code":"\\n\\n    var angle = cb_obj.value*Math.PI/180;\\n    sx.gradient=Math.tan(angle);\\n    sy.gradient=Math.tan(angle + Math.PI/2);\\n    var c00 = 1;\\n    var c10 = 2;\\n    var c11 = 5;\\n\\n    var ca = Math.cos(-angle);\\n    var sa = Math.sin(-angle);\\n    var ca2 = ca*ca;\\n    var sa2 = sa*sa;\\n    var sca = sa*ca;\\n\\n    var c00r = ca2*c00 + sa2*c11 - 2*sca*c10;\\n    var c01r = sca*(c00 - c11) + (ca2 - sa2)*c10;\\n    var c10r = c01r;\\n    var c11r = sa2*c00 + ca2*c11 + 2*sca*c10;\\n\\n    source.data[&#x27;c1&#x27;] = [c00r, c10r];\\n    source.data[&#x27;c2&#x27;] = [c01r, c11r];\\n    source.change.emit();\\n"},"id":"3153","type":"CustomJS"},{"attributes":{"axis":{"id":"3117"},"ticker":null},"id":"3120","type":"Grid"},{"attributes":{"axis":{"id":"3121"},"dimension":1,"ticker":null},"id":"3124","type":"Grid"},{"attributes":{"gradient":0.0,"line_color":"orange","line_dash":[6],"line_width":3.5,"y_intercept":0},"id":"3150","type":"Slope"},{"attributes":{"data_source":{"id":"3139"},"glyph":{"id":"3140"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3141"},"view":{"id":"3143"}},"id":"3142","type":"GlyphRenderer"},{"attributes":{},"id":"3167","type":"AllLabels"},{"attributes":{"formatter":{"id":"3168"},"major_label_policy":{"id":"3170"},"ticker":{"id":"3118"}},"id":"3117","type":"LinearAxis"},{"attributes":{"below":[{"id":"3117"}],"center":[{"id":"3120"},{"id":"3124"},{"id":"3150"},{"id":"3151"}],"left":[{"id":"3121"}],"match_aspect":true,"renderers":[{"id":"3142"}],"title":{"id":"3157"},"toolbar":{"id":"3132"},"x_range":{"id":"3109"},"x_scale":{"id":"3113"},"y_range":{"id":"3111"},"y_scale":{"id":"3115"}},"id":"3108","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3160","type":"Selection"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"3131","type":"BoxAnnotation"},{"attributes":{"overlay":{"id":"3131"}},"id":"3127","type":"BoxZoomTool"},{"attributes":{"source":{"id":"3139"}},"id":"3143","type":"CDSView"}],"root_ids":["3156"]},"title":"Bokeh Application","version":"2.3.1"}}';
                  var render_items = [{"docid":"d4511a28-9eaf-4b1a-abe0-ebee0872d6cb","root_ids":["3156"],"roots":{"3156":"65f7bd09-d4f1-469e-b724-4de2189c2e59"}}];
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