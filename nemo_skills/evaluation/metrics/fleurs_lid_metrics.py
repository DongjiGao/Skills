# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top-1 FLEURS-LID metrics with locale and geographic reporting."""

from nemo_skills.evaluation.metrics.language_id_metrics import LanguageIDMetrics


class FleursLIDMetrics(LanguageIDMetrics):
    def metrics_to_print(self):
        metrics = super().metrics_to_print()
        metrics["num_locales"] = metrics.pop("num_labels")
        return metrics

    def get_metrics(self):
        metrics = super().get_metrics()
        for values in metrics.values():
            values["num_locales"] = values.pop("num_labels")
        return metrics
