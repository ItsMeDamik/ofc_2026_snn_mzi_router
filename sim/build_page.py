"""Insert sim_data.json into the page template -> spike_express_lanes.html (open in a browser)."""
t = open("page_template.html").read()
d = open("sim_data.json").read()
open("spike_express_lanes.html", "w").write(t.replace("__DATA__", d))
print("wrote spike_express_lanes.html")
