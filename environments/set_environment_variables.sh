#!/bin/sh

# set $LD_LIBRARY to avoid reverb libpython errors

cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh

# Edit ./etc/conda/activate.d/env_vars.sh as follows:
FILE='./etc/conda/activate.d/env_vars.sh'
grep -qxF '#!/bin/sh' "$FILE" || echo '#!/bin/sh' >> "$FILE"
LINE1='export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}'
LINE2="export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/:"
grep -qxF "$LINE1" "$FILE" || echo  "$LINE1" >> "$FILE"
grep -qxF "$LINE2" "$FILE" || echo  "$LINE2"'${LD_LIBRARY_PATH}' >> "$FILE"


# Edit ./etc/conda/deactivate.d/env_vars.sh as follows:
FILE='./etc/conda/deactivate.d/env_vars.sh'
grep -qxF '#!/bin/sh' "$FILE" || echo '#!/bin/sh' >> "$FILE"
LINE1='export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}'
LINE2='unset OLD_LD_LIBRARY_PATH'
grep -qxF "$LINE1" "$FILE" || echo  "$LINE1" >> "$FILE"
grep -qxF "$LINE2" "$FILE" || echo  "$LINE2" >> "$FILE"