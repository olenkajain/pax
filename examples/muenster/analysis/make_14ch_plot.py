__author__ = 'axel'
import matplotlib.pyplot as plt
plt.rc('font', size=16)

def make_plot(hits_df, field, xaxislabel = "x-axis", yaxislabel = "y-axis", nof_bins=500, xlim=(-1,5000), ylim= None, legendlabel=None, **kwargs):

    fig, ax = plt.subplots(nrows=7, ncols=2, sharex=True, sharey=True, squeeze=False, figsize=(20,14))
    plt.subplots_adjust(hspace=0.1, wspace=0.05)

    ax[6][0].set_xlabel(xaxislabel)
    ax[6][1].set_xlabel(xaxislabel)
    for xi in range(7):
            ax[xi][0].set_xlim(xlim)
            if ylim:
                ax[xi][0].set_ylim(ylim)
            ax[xi][0].set_ylabel(yaxislabel)
            ax[xi][0].set_yscale("log")
            ax[xi][0].hist([xhits[field] for xhits in hits_df if (xhits["channel"] == xi)],
                                #range=(0,10),
                                bins=nof_bins,
                                histtype="step",
                                label=legendlabel,
                                **kwargs)
            ax[xi][0].text(8, 1000, "Channel # %i"%xi)
    for xi in range(7, 14):
            ax[xi-7][1].set_yscale("log")
            #ax[xi][yi].set_ylim(0,10000)
            ax[xi-7][1].hist([xhits[field] for xhits in hits_df if (xhits["channel"] == xi)],
                                #range=(0,10),
                                bins=nof_bins,
                                histtype="step",
                                label=legendlabel,
                                **kwargs)
            ax[0][1].legend(framealpha=0.5)
            ax[xi-7][1].text(8, 1000, "Channel # %i"%xi)
    return fig
