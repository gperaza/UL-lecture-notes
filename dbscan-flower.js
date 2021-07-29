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
    
      
      
    
      var element = document.getElementById("f8c362b4-1540-46f9-bfd4-63850f1dae37");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'f8c362b4-1540-46f9-bfd4-63850f1dae37' but no matching script tag was found.")
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
                    
                  var docs_json = '{"55d7e167-10a1-4a9f-9c60-9b9cb5b5acdb":{"defs":[],"roots":{"references":[{"attributes":{"children":[{"id":"1518"}]},"id":"1553","type":"Row"},{"attributes":{"children":[{"id":"1550"},{"id":"1551"}]},"id":"1552","type":"Row"},{"attributes":{},"id":"1561","type":"AllLabels"},{"attributes":{},"id":"1560","type":"BasicTickFormatter"},{"attributes":{},"id":"1521","type":"DataRange1d"},{"attributes":{"end":50,"start":5,"step":5,"title":"eps","value":25},"id":"1550","type":"Slider"},{"attributes":{"formatter":{"id":"1557"},"major_label_policy":{"id":"1558"},"ticker":{"id":"1528"}},"id":"1527","type":"LinearAxis"},{"attributes":{},"id":"1525","type":"LinearScale"},{"attributes":{"axis":{"id":"1531"},"dimension":1,"ticker":null},"id":"1534","type":"Grid"},{"attributes":{},"id":"1536","type":"WheelZoomTool"},{"attributes":{},"id":"1555","type":"Title"},{"attributes":{"active_multi":null,"tools":[{"id":"1535"},{"id":"1536"},{"id":"1537"},{"id":"1538"},{"id":"1539"},{"id":"1540"}]},"id":"1542","type":"Toolbar"},{"attributes":{"end":200,"start":20,"step":10,"title":"ms","value":100},"id":"1551","type":"Slider"},{"attributes":{"formatter":{"id":"1560"},"major_label_policy":{"id":"1561"},"ticker":{"id":"1532"}},"id":"1531","type":"LinearAxis"},{"attributes":{},"id":"1523","type":"LinearScale"},{"attributes":{},"id":"1538","type":"SaveTool"},{"attributes":{"below":[{"id":"1527"}],"center":[{"id":"1530"},{"id":"1534"}],"left":[{"id":"1531"}],"match_aspect":true,"title":{"id":"1555"},"toolbar":{"id":"1542"},"x_range":{"id":"1519"},"x_scale":{"id":"1523"},"y_range":{"id":"1521"},"y_scale":{"id":"1525"}},"id":"1518","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"1558","type":"AllLabels"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"1541","type":"BoxAnnotation"},{"attributes":{},"id":"1535","type":"PanTool"},{"attributes":{},"id":"1539","type":"ResetTool"},{"attributes":{},"id":"1540","type":"HelpTool"},{"attributes":{},"id":"1532","type":"BasicTicker"},{"attributes":{},"id":"1528","type":"BasicTicker"},{"attributes":{"children":[{"id":"1552"},{"id":"1553"}]},"id":"1554","type":"Column"},{"attributes":{},"id":"1519","type":"DataRange1d"},{"attributes":{"axis":{"id":"1527"},"ticker":null},"id":"1530","type":"Grid"},{"attributes":{"overlay":{"id":"1541"}},"id":"1537","type":"BoxZoomTool"},{"attributes":{},"id":"1557","type":"BasicTickFormatter"}],"root_ids":["1554"]},"title":"Bokeh Application","version":"2.3.1"}}';
                  var render_items = [{"docid":"55d7e167-10a1-4a9f-9c60-9b9cb5b5acdb","root_ids":["1554"],"roots":{"1554":"f8c362b4-1540-46f9-bfd4-63850f1dae37"}}];
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