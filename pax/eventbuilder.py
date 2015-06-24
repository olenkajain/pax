"""Event builder routines

The event builder is responsible for finding events within the DAQ output
stream and turning this into events that the end user can process.  The data is
fetched untriggered, which means that there is no event structure.  Or rather,
it is just a stream of PMT pulses that need to be assembled into events.  This
is exactly what this code does.  There are two inputs requires:

* Location of runs database
* Trigger settings

This code is meant to run in production, therefore a runs database must be
 provided.  This code will use this database to locate what data needs to be
 processed and handle it appropriately.

This code is also responsible triggering (i.e., deciding which events to
record).  At present there are two stages to the triggering: a pretrigger and
a pax-event trigger.  The pretrigger is a simple sliding window coincidence
trigger that runs over the data to identify possible signals.  The parameters
of this are then, clearly, the size of the window and the coincidence
requirement.  Another parameter is how much data to save around the discovered
peak, which often is an S2.

The second level trigger, which we often do not run, processed the event in pax
and is used to make, e.g., radial cuts.  At present, the code just writes HDF5
files.

It is worth mentioning that is an 'expert modes' with a flag for mega events:
an event of one second is made that can be used for computing the trigger
efficiency.

The entry point to this code is typically via bin/event-builder.
"""

import argparse
import logging
import pymongo
import time
from pax import core, units


def run():
    """Start running the event builder

    Find a dataset to process, then process it with settings from command line.
    """
    args, log = handle_args()

    query = {"trigger.status": "waiting_to_be_processed"}

    log.info("Searching for run")

    client = pymongo.MongoClient(args.address,
                                 args.port,
                                 serverSelectionTimeoutMS=500)
    try:
        client.admin.command('ping')
        log.debug("Connection successful to %s:%d",
                  args.address,
                  args.port)
    except pymongo.errors.ConnectionFailure:
        log.fatal("Cannot connect to MongoDB at %s:%d" % (args.address,
                                                          args.port))
        raise

    log.debug('Fetching databases: %s', args.database)
    db = client.get_database(args.database)

    log.debug('Getting collection: %s', args.collection)
    collection = db.get_collection(args.collection)

    while 1:
        run_doc = collection.find_one_and_update(query,
                                                 {'$set': {'trigger.status': 'staging'}})

        if run_doc is None:
            if args.impatient:
                log.info("Too impatient to wait for data, exiting...")
                break
            else:
                log.info("No data to process... waiting %d seconds",
                         args.wait)
                time.sleep(args.wait)
        else:
            log.info("Building events for %s",
                     run_doc['name'])

            filename = '%s' % run_doc['name']

            if args.processed:
                plugin_group_names = ['input',  'preprocessing',  'dsp',
                                      'transform', 'output']
                output = ['BulkOutput.BulkOutput', 'BulkOutput.BulkOutput']
            else:
                plugin_group_names = ['input',  'preprocessing', 'output']
                output = ['BSON.WriteZippedBSON']

            config_names = 'eventbuilder'
            config_dict = {'DEFAULT': {'run_doc': run_doc['_id']},
                           'pax': {'plugin_group_names': plugin_group_names,
                                   'output': output,
                                   'output_name': filename, },

                           'MongoDB': {'runs_database_location': {'address': args.address,
                                                                  'database': args.database,
                                                                  'port': args.port,
                                                                  'collection': args.collection
                                                                  },
                                       'window': args.window * units.us,
                                       'left': args.left * units.us,
                                       'right': args.right * units.us,
                                       'multiplicity': args.multiplicity,
                                       'mega_event': args.mega_event
                                       }
                           }
            try:
                p = core.Processor(config_names=config_names,
                                   config_dict=config_dict)

                p.run()

            except pymongo.errors.ServerSelectionTimeoutError as e:
                log.exception(e)
                collection.update(query,
                                  {'$set': {'trigger.status': 'error'}})


def handle_args():
    """Command line argument processing

    This routine is also responsible for setting up logging.
    """
    parser = argparse.ArgumentParser(description="Build XENON1T events from the"
                                                 " data aquisiton. This tools "
                                                 "starts the distributed "
                                                 "processing of events.")
    parser.add_argument('--impatient',
                        action='store_true',
                        help="Event builder will not wait for new data")
    parser.add_argument('--processed',
                        action='store_true',
                        help="Write processed files too")
    parser.add_argument('--mega_event',
                        action='store_true',
                        help="used for trigger efficiency")
    trigger_group = parser.add_argument_group(title='Event builder settings',
                                              description='Configure trigger')
    trigger_group.add_argument('--multiplicity',
                               type=int,
                               help='Number pulses required for coincidence '
                                    'trigger',
                               default=10)
    trigger_group.add_argument('--window',
                               type=int,
                               help='Size of sliding window (us)',
                               default=1)
    trigger_group.add_argument('--left',
                               type=int, default=200,
                               help='Left extension to save (us)')
    trigger_group.add_argument('--right',
                               type=int, default=200,
                               help='Right extension to save (us)')
    run_db_str = 'The runs database stores all metadata about runs, including' \
                 ' which are waiting to be triggered. Communication is ' \
                 'required to find data.  More information on the MongoDB ' \
                 'jargon (e.g., "collection") can be found in their docs.'
    run_db_group = parser.add_argument_group(title='Runs database settings',
                                             description=run_db_str)
    run_db_group.add_argument('--address',
                              default='daqeb0',
                              help='Address or hostname of MongoDB instance.')
    run_db_group.add_argument('--database',
                              default='online',
                              help='')
    run_db_group.add_argument('--collection',
                              default='runs',
                              help='')
    run_db_group.add_argument('--port',
                              default=27000,
                              type=int,
                              help='Listening port of MongoDB.')
    parser.add_argument('--wait',
                        default=1,
                        type=int,
                        help="Wait time between searching if no data")
    # Log level control
    parser.add_argument('--log', default=None,
                        help="Set log level, e.g. 'debug'")
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)s %(levelname)-8s %('
                               'message)s',
                        datefmt='%m-%d %H:%M',
                        filename='myapp.log',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    log = logging.getLogger('eb')
    args = parser.parse_args()
    return args, log
