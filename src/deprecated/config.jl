MATDIR = try
    ENV["MATDIR"]
catch
    "~/"
end
DEFSAVEDIR = try
    ENV["DEFSAVEDIR"]
catch
    "~/"
end
DEFCONVTHRESH = 10^-3
DEFLCTHRESH = 10^-3 # smaller threshold means faster lightcone
DEFPREC = 10^-5
DEFDLIM = 100


