# coding=utf-8
#
# Copyright 2020-2022 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
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

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import (BertModel, BertPreTrainedModel)

PARENT_CLASSES = {
    'bert': BertPreTrainedModel
}

MODEL_CLASSES = {
    BertPreTrainedModel: BertModel
}


def TransformerForDST(parent_name):
    if parent_name not in PARENT_CLASSES:
        raise ValueError("Unknown model %s" % (parent_name))

    class TransformerForDST(PARENT_CLASSES[parent_name]):
        def __init__(self, config):
            assert config.model_type in PARENT_CLASSES
            assert self.__class__.__bases__[0] in MODEL_CLASSES
            super(TransformerForDST, self).__init__(config)
            self.model_type = config.model_type
            self.slot_list = config.dst_slot_list
            self.class_types = config.dst_class_types
            self.class_labels = config.dst_class_labels
            self.token_loss_for_nonpointable = config.dst_token_loss_for_nonpointable
            self.refer_loss_for_nonpointable = config.dst_refer_loss_for_nonpointable
            self.class_aux_feats_inform = config.dst_class_aux_feats_inform
            self.class_aux_feats_ds = config.dst_class_aux_feats_ds
            self.class_loss_ratio = config.dst_class_loss_ratio

            # Only use refer loss if refer class is present in dataset.
            if 'refer' in self.class_types:
                self.refer_index = self.class_types.index('refer')
            else:
                self.refer_index = -1

            # Make sure this module has the same name as in the pretrained checkpoint you want to load!
            self.add_module(self.model_type, MODEL_CLASSES[self.__class__.__bases__[0]](config))
            
            self.dropout = nn.Dropout(config.dst_dropout_rate)
            self.dropout_heads = nn.Dropout(config.dst_heads_dropout_rate)

            if self.class_aux_feats_inform:
                self.add_module("inform_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))
            if self.class_aux_feats_ds:
                self.add_module("ds_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))

            aux_dims = len(self.slot_list) * (self.class_aux_feats_inform + self.class_aux_feats_ds) # second term is 0, 1 or 2

            #######################################################################################################
            ## TODO DST_TASK1 TripPy gates
            ## Here we define the gates used to determine the source of value copying. 
            ## An example for refer gate is given
            ## Start of your code #################################################################################

            for slot in self.slot_list:
                # Class gate: uses CLS token embedding plus aux features, predicts one of the class types
                self.add_module("class_" + slot, nn.Linear(config.hidden_size + aux_dims, len(self.class_types)))
                # Span gate: uses token embeddings and predicts start/end logits at once for each token
                self.add_module("span_" + slot, nn.Linear(config.hidden_size, 2))
                # Refer gate: uses CLS token embedding plus aux features, predicts referring slot or none
                self.add_module("refer_" + slot, nn.Linear(config.hidden_size + aux_dims, len(self.slot_list) + 1))

            ## End of your code ###################################################################################
            
            # Head for aux task
            if hasattr(config, "aux_task_def") and config.aux_task_def is not None:
                self.add_module("aux_out_projection", nn.Linear(config.hidden_size, int(config.aux_task_def['n_class'])))

            self.init_weights()

        def forward(self,
                    input_ids,
                    input_mask=None,
                    segment_ids=None,
                    position_ids=None,
                    head_mask=None,
                    start_pos=None,
                    end_pos=None,
                    inform_slot_id=None,
                    refer_id=None,
                    class_label_id=None,
                    diag_state=None,
                    aux_task_def=None):
            outputs = getattr(self, self.model_type)(
                input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                position_ids=position_ids,
                head_mask=head_mask
            )

            sequence_output = outputs[0]
            pooled_output = outputs[1]

            sequence_output = self.dropout(sequence_output)
            pooled_output = self.dropout(pooled_output)

            if aux_task_def is not None:
                if aux_task_def['task_type'] == "classification":
                    aux_logits = getattr(self, 'aux_out_projection')(pooled_output)
                    aux_logits = self.dropout_heads(aux_logits)
                    aux_loss_fct = CrossEntropyLoss()
                    aux_loss = aux_loss_fct(aux_logits, class_label_id)
                    # add hidden states and attention if they are here
                    return (aux_loss,) + outputs[2:]
                elif aux_task_def['task_type'] == "span":
                    aux_logits = getattr(self, 'aux_out_projection')(sequence_output)
                    aux_start_logits, aux_end_logits = aux_logits.split(1, dim=-1)
                    aux_start_logits = self.dropout_heads(aux_start_logits)
                    aux_end_logits = self.dropout_heads(aux_end_logits)
                    aux_start_logits = aux_start_logits.squeeze(-1)
                    aux_end_logits = aux_end_logits.squeeze(-1)

                    # If we are on multi-GPU, split add a dimension
                    if len(start_pos.size()) > 1:
                        start_pos = start_pos.squeeze(-1)
                    if len(end_pos.size()) > 1:
                        end_pos = end_pos.squeeze(-1)
                    # sometimes the start/end positions are outside our model inputs, we ignore these terms
                    ignored_index = aux_start_logits.size(1) # This is a single index
                    start_pos.clamp_(0, ignored_index)
                    end_pos.clamp_(0, ignored_index)

                    aux_token_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                    aux_start_loss = aux_token_loss_fct(torch.cat((aux_start_logits, aux_end_logits), 1), start_pos)
                    aux_end_loss = aux_token_loss_fct(torch.cat((aux_end_logits, aux_start_logits), 1), end_pos)
                    aux_loss = (aux_start_loss + aux_end_loss) / 2.0
                    return (aux_loss,) + outputs[2:]
                else:
                    raise ValueError("Unknown task_type %s" % (aux_task_def['task_type'] if 'task_type' in aux_task_def else None))

            if inform_slot_id is not None:
                inform_labels = torch.stack(list(inform_slot_id.values()), 1).float()
            if diag_state is not None:
                diag_state_labels = torch.clamp(torch.stack(list(diag_state.values()), 1).float(), 0.0, 1.0)


            #######################################################################################################
            ## TODO DST_TASK2 Prediction and loss computation
            ## Complete the following section where it is marked with TODO (2a-2f)
            ## Start of your code #################################################################################

            total_loss = 0
            per_slot_per_example_loss = {}
            per_slot_class_logits = {}
            per_slot_start_logits = {}
            per_slot_end_logits = {}
            per_slot_refer_logits = {}
            for slot in self.slot_list:
                if self.class_aux_feats_inform and self.class_aux_feats_ds:
                    pooled_output_aux = torch.cat((pooled_output, self.inform_projection(inform_labels), self.ds_projection(diag_state_labels)), 1)
                elif self.class_aux_feats_inform:
                    pooled_output_aux = torch.cat((pooled_output, self.inform_projection(inform_labels)), 1)
                elif self.class_aux_feats_ds:
                    pooled_output_aux = torch.cat((pooled_output, self.ds_projection(diag_state_labels)), 1)
                else:
                    pooled_output_aux = pooled_output

                #######################################################################################################
                ## TODO 2a obtain logits for slot class type prediction. Class prediction uses CLS token embedding 
                ## After obtaining the logits, you can also pass them to self.dropout_heads to encourage the model to learn more generalized features

                # Class logits: use pooled CLS embedding (+ aux features when enabled)
                class_logits = getattr(self, "class_" + slot)(pooled_output_aux)
                class_logits = self.dropout_heads(class_logits)

                #######################################################################################################

                #######################################################################################################
                ## TODO 2b obtain token_logits for slot span prediction. Span prediction uses embedding of each token
                ## After obtaining the logits, you can also pass them to self.dropout_heads to encourage the model to learn more generalized features

                # token_logits will contain both the start and end logits, split them and assign them accordingly, pay attention to the shape of the tensor 
                # Span logits: use token-level embeddings to predict start and end logits
                token_logits = getattr(self, "span_" + slot)(sequence_output)
                # token_logits: (batch, seq_len, 2) -> split into start/end (batch, seq_len, 1)
                start_logits, end_logits = token_logits.split(1, dim=-1)
                start_logits = self.dropout_heads(start_logits).squeeze(-1)
                end_logits = self.dropout_heads(end_logits).squeeze(-1)

                #######################################################################################################

                #######################################################################################################
                ## TODO 2c obtain logits for slot referral prediction. Refer prediction uses CLS token embedding 
                ## After obtaining the logits, you can also pass them to self.dropout_heads to encourage the model to learn more generalized features

                # Refer logits: use pooled CLS embedding (+ aux features when enabled)
                refer_logits = getattr(self, "refer_" + slot)(pooled_output_aux)
                refer_logits = self.dropout_heads(refer_logits)

                #######################################################################################################

                per_slot_class_logits[slot] = class_logits
                per_slot_start_logits[slot] = start_logits
                per_slot_end_logits[slot] = end_logits
                per_slot_refer_logits[slot] = refer_logits

                # If there are no labels, don't compute loss
                if class_label_id is not None and start_pos is not None and end_pos is not None and refer_id is not None:
                    # If we are on multi-GPU, split add a dimension
                    if len(start_pos[slot].size()) > 1:
                        start_pos[slot] = start_pos[slot].squeeze(-1)
                    if len(end_pos[slot].size()) > 1:
                        end_pos[slot] = end_pos[slot].squeeze(-1)
                    # sometimes the start/end positions are outside our model inputs, we ignore these terms
                    ignored_index = start_logits.size(1) # This is a single index
                    start_pos[slot].clamp_(0, ignored_index)
                    end_pos[slot].clamp_(0, ignored_index)

                    class_loss_fct = CrossEntropyLoss(reduction='none')
                    token_loss_fct = CrossEntropyLoss(reduction='none', ignore_index=ignored_index)
                    refer_loss_fct = CrossEntropyLoss(reduction='none')

                    #######################################################################################################
                    ## TODO 2d Compute loss for slot class type prediction 

                    # Compute token (span) loss. We follow the same pattern as the auxiliary span task
                    # start_logits/end_logits: (batch, seq_len)
                    start_loss = token_loss_fct(torch.cat((start_logits, end_logits), 1), start_pos[slot])
                    end_loss = token_loss_fct(torch.cat((end_logits, start_logits), 1), end_pos[slot])

                    token_loss = (start_loss + end_loss) / 2.0
                    #######################################################################################################

                    token_is_pointable = (start_pos[slot] > 0).float()
                    if not self.token_loss_for_nonpointable:
                        token_loss *= token_is_pointable

                    refer_loss = refer_loss_fct(refer_logits, refer_id[slot])
                    token_is_referrable = torch.eq(class_label_id[slot], self.refer_index).float()
                    if not self.refer_loss_for_nonpointable:
                        refer_loss *= token_is_referrable

                    #######################################################################################################
                    ## TODO 2e Compute loss for slot class type prediction 

                    # Class loss (per-example)
                    class_loss = class_loss_fct(class_logits, class_label_id[slot])

                    #######################################################################################################


                    #######################################################################################################
                    ## TODO 2f Compute final loss from all predictions
                    ## see exercise sheet for the loss formula

                    # Combine losses according to weighting hyperparameter
                    if self.refer_index > -1:
                        # include refer loss
                        per_example_loss = self.class_loss_ratio * class_loss + \
                                           (1.0 - self.class_loss_ratio) / 2.0 * token_loss + \
                                           (1.0 - self.class_loss_ratio) / 2.0 * refer_loss
                    else:
                        # no refer class in dataset -> only class + token loss
                        per_example_loss = self.class_loss_ratio * class_loss + \
                                           (1.0 - self.class_loss_ratio) * token_loss

                    per_slot_per_example_loss[slot] = per_example_loss

            # Aggregate total loss: sum mean loss across slots
            total_loss = 0
            if len(self.slot_list) > 0:
                slot_losses = [per_slot_per_example_loss[s].mean() for s in self.slot_list]
                total_loss = torch.stack(slot_losses, 0).sum()
                    #######################################################################################################

            # add hidden states and attention if they are here
            outputs = (total_loss,) + (per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits, per_slot_refer_logits,) + outputs[2:]


            return outputs

    return TransformerForDST
